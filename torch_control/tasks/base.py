import abc
import collections
import copy
from math import ceil
from typing import Callable, Dict, Type, Union

import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from omegaconf import ListConfig
from torch_control.predictive_models import RigidBody
from torch_control.dynamics.quadrotor.quadrotor import Quadrotor
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.utils import rot_utils as ru
from torch_control.utils.common import HistoryQueue, nan_alarm
from torch_control.utils.butterworth_lpf import ButterworthLowPassFilterBatchedParallel as LPF

def to_tensor(value, device):
    # Converts value to a tensor if it is not None while handling ListConfig
    if isinstance(value, (ListConfig, list)):
        value = torch.tensor(list(value), dtype=torch.float32, device=device)
    elif isinstance(value, (int, float)):
        value = torch.tensor(value, dtype=torch.float32, device=device)
    return value


class GymEnv(gym.Env):

    REGISTRY: Dict[str, Type['GymEnv']] = {}

    def __init__(self, cfg, num_envs: int = None):
        self.cfg = cfg
        if cfg.get('for_deploy', False):
            self.task_cfg = cfg.task_deploy
        else:
            self.task_cfg = cfg.task
        self.quadrotor_cfg = cfg.quadrotor
        self.obs_cfg = cfg.controller.obs
        self.action_cfg = cfg.task.action
        self.reward_cfg = cfg.reward

        # basic settings
        self.num_envs = num_envs if num_envs is not None else cfg.num_envs
        self.device = self.cfg.device
        self.auto_reset = self.cfg.auto_reset
        self.early_termination = self.cfg.early_termination

        self.robots = Quadrotor(self.quadrotor_cfg, self.num_envs, self.device,
                                wind_config=self.cfg.wind,
                                linear_std=self.cfg.noise.linear_std,
                                angular_std=self.cfg.noise.angular_std,)
        
        self.predictive_model = RigidBody(dt=self.robots.control_dt)
        self.pred_model_noise = self.cfg.pred_model_noise
        self.use_rigid_body = True
        self.no_random_steps = 0

        # task settings
        self.max_time = self.task_cfg.max_time
        self.max_episode_length = int(self.max_time * self.robots.control_freq)
        self.elapsed_steps = 0
    
        # randomization
        self.random_dynamics = self.cfg.random_dynamics
        self.random_init = self.cfg.random_init
        
        # safety boundaries
        self.boundary_offset = torch.as_tensor(np.array([
            list(self.task_cfg.safety.x), 
            list(self.task_cfg.safety.y), 
            list(self.task_cfg.safety.z)]), dtype=torch.float32, device=self.device)

        self.setup_randomizations()
        self.init_vars()
        self.init_history()
        self.setup_specs()

        if self.cfg.get('action_lpf', False) and self.cfg.action_lpf.enable:
            self.action_lpf = LPF(self.cfg.action_lpf.cutoff_freq,
                                  self.robots.control_dt,
                                  batch_size=self.num_envs,
                                  action_size=self.action_size,
                                  order=self.cfg.action_lpf.order,
                                  device=self.device)
        else:
            self.action_lpf = None

        print("[Env.action_lpf] ", self.cfg.action_lpf)
        
    def get_robot_pos(self, idx):
        pass
    
    def get_robot_quat(self, idx):
        pass
        
    def set_predictive_model(self, model: nn.Module, no_random_steps: int = 0):
        self.predictive_model = copy.deepcopy(model)
        self.predictive_model.reset_aux_vars(self.num_envs)
        if self.use_rigid_body:
            print("[Env.PredictiveModel] Set predictive model:", model.__repr__())
        self.use_rigid_body = False
        self.no_random_steps = no_random_steps

    @torch.no_grad()
    def update_predictive_model(self, model: nn.Module, alpha: float = 1.0):
        for p, p_new in zip(self.predictive_model.parameters(), model.parameters()):
            p.data = (1 - alpha) * p.data + alpha * p_new.data

    def reset(self, 
              idx: torch.Tensor = None, 
              state_vec: torch.Tensor = None, 
              task_vec: torch.Tensor = None) -> torch.Tensor:
        if idx is None:
            idx = torch.arange(self.num_envs)

        self.elapsed_steps += torch.sum(self.progress[idx])

        if self.no_random_steps > 0 and self.elapsed_steps < self.no_random_steps:
            dynamics_randomization = {}
        else:
            dynamics_randomization = self.dynamics_randomization

        self.robots.reset(idx, 
                          self.state_randomization, 
                          dynamics_randomization, 
                          state_vec=state_vec)
        self.reset_task(idx, task_vec)
        
        self.state = self.robots.state.clone()
        self.state[idx] = self.robots.state[idx].clone()
        self.last_state[idx] = self.state[idx].clone()
        self.boundaries[idx] = self.state[idx].pos.unsqueeze(-1) + self.boundary_offset
        
        self.cmd[idx] = 0.
        self.time[idx] = 0.
        self.acc_reward[idx] = 0.
        self.progress[idx] = 0.
        self.last_cmd[idx] = self.cmd[idx].clone()

        if hasattr(self, 'history_buffer'):
            for name, history in self.history_buffer.items():
                if name == 'state':
                    init_state_obs = self.get_state_obs(self.state[idx])
                    history.reset(idx, init_state_obs)
                else:
                    history.reset(idx)

        if not self.use_rigid_body:
            self.predictive_model.reset_aux_vars(idx)

        if self.action_lpf is not None:
            self.action_lpf.reset(idx)
                
        for m in self.eval_metrics:
            if hasattr(self, f'record_{m.split("(")[0]}'):
                getattr(self, f'record_{m.split("(")[0]}')[idx] = 0.
        
        obs = self.get_obs(self.state, self.last_state, self.cmd, self.time)
        
        if self.cfg.latency_obs > 0:
            self.obs_buffer.reset(idx)
            self.action_buffer.reset(idx)
            obs = self.obs_buffer.update(obs)
            
        # clean gpu memory
        torch.cuda.empty_cache()    
        
        return obs
    
    @abc.abstractmethod
    def reset_task(self, idx: torch.Tensor = None, task_vec: torch.Tensor = None):
        """
        Reset the task-specific parameters.
        """
        raise NotImplementedError
    
    def step(self, actions):
        with torch.no_grad():
            self.last_state = self.state.clone()
            self.last_cmd = self.cmd.clone()
            
        if self.cfg.latency_cmd > 0:
            actions = self.action_buffer.update(actions)

        self.cmd = actions.clone()

        self.state, dt = self.robots.step(self.process_action(actions))
        self.time += dt
        self.progress += 1
        
        obs = self.get_obs(self.state, self.last_state, self.cmd, self.time)

        all_rewards = self.get_reward(self.state, self.last_state, self.cmd, self.last_cmd, self.time)
            
        reward = torch.sum(torch.stack(list(all_rewards.values())), dim=0)
        
        self.acc_reward = self.acc_reward + reward
        
        is_safe = self.check_safety_constraints(self.state)
        truncation = self.check_timeout()
        if self.early_termination:
            done = truncation | (~is_safe)
        else:
            done = truncation
        
        info = {
            'is_safe': is_safe,
            'truncation': truncation,
            'episode_return': self.acc_reward.clone(),
            'episode_length': self.progress.clone(),
        }

        all_rewards['reward_sum'] = reward.clone()
        
        metrics = self.update_metrics(all_rewards, self.state, self.last_state, self.cmd, self.time)
        for m in metrics:
            assert torch.isnan(metrics[m]).any() == False, f"NaN detected in {m}"
        info.update({
            f'episode_{m}': metrics[m] * done.float()
            if 'traj' not in m else metrics[m] * done.float().unsqueeze(-1)
            for m in metrics})

        if self.auto_reset:
            done_idx = done.nonzero(as_tuple=True)[0]
            if len(done_idx) > 0:
                info['obs_before_reset'] = obs.clone()
                obs = self.reset(done_idx)
                
        info.update({'progress': self.progress.clone()})
                
        return obs, reward, done, info
    
    def eval_step(self, actions, 
                  state: QuadrotorState=None, 
                  dt: float=None, 
                  update: bool=True):
        if state is None:
            state = self.state
        
        state, dt = self.robots.step(actions, dt=dt)
        self.time = self.time + dt
        self.progress = self.progress + 1

        is_safe = self.check_safety_constraints(state)
        truncation = self.check_timeout()
        if self.early_termination:
            done = truncation | (~is_safe)
        else:
            done = truncation
            
        if update:
            self.state = state
        
        return state, done

    def check_timeout(self):
        timeout = self.progress >= self.max_episode_length
        return timeout

    def check_safety_constraints(self, state: QuadrotorState = None):
        if state is None:
            state = self.state
            
        is_safe = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        is_safe &= torch.all(state.pos >= self.boundaries[..., 0], dim=-1)
        is_safe &= torch.all(state.pos <= self.boundaries[..., 1], dim=-1)
        return is_safe

    def clear_grad(self):
        self.robots.clear_grad()
        
        if hasattr(self, 'history_buffer'):
            for history in self.history_buffer.values():
                history.clear_grad()
        
        all_attrs = [attr for attr in dir(self) 
                     if isinstance(getattr(self, attr), torch.Tensor)]
        for attr in all_attrs:
            setattr(self, attr, getattr(self, attr).clone())
            
    ########################## Utils ##########################
    def process_action(self, action):
        if self.action_lpf is not None:
            action = self.action_lpf.filter(action)

        action = torch.clamp(action,
                             min=self.action_range[..., 0],
                             max=self.action_range[..., 1])
        if self.action_cfg.offset_g:
            if self.robots.control_mode == 'ctbr':
                action[..., 0] += self.robots.g
            elif self.robots.control_mode == 'srt':
                action[..., :] += self.robots.g / 4
            else:
                raise NotImplementedError(f'Invalid control mode: {self.robots.control_mode}')
        return action

    def get_reward(self, 
                   state: QuadrotorState = None,
                   last_state: QuadrotorState = None,
                   cmd: torch.Tensor = None,
                   last_cmd: torch.Tensor = None,
                   time: torch.Tensor = None) -> Dict:
        if state is None:
            state = self.state
        if last_state is None:
            last_state = self.last_state
        if cmd is None:
            cmd = self.cmd
        if last_cmd is None:
            last_cmd = self.last_cmd
        if time is None:
            time = self.time

        rewards = {}
    
        for term in self.reward_cfg.smooth:
            weight = self.reward_cfg.smooth.get(term)
            if weight is None or weight <= 0:
                continue
            if term in QuadrotorState.full_state:
                diff = torch.norm(state.get(term) - last_state.get(term), dim=-1)
                rewards[f'reward_smooth_{term}'] = weight * torch.exp(-diff)
            elif term == 'cmd':
                diff = torch.norm(cmd - last_cmd, dim=-1)
                rewards['reward_smooth_cmd'] = weight * torch.exp(-diff)
            else:
                raise KeyError
        
        for term in self.reward_cfg.min:
            weight = self.reward_cfg.min.get(term)
            if weight is None:
                continue
            if term in QuadrotorState.full_state:
                scale = torch.norm(state.get(term), dim=-1)
                rewards[f'reward_min_{term}'] = weight * torch.exp(-scale)
            elif term == 'cmd':
                scale = torch.norm(cmd, dim=-1)
                rewards['reward_min_cmd'] = weight * torch.exp(-scale)
            else:
                raise KeyError
            
        if self.reward_cfg.alive is not None and self.reward_cfg.alive > 0:
            rewards['reward_alive'] = self.reward_cfg.alive * torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        
        return rewards

    def get_obs(self, 
                state: QuadrotorState = None, 
                last_state: QuadrotorState = None,
                cmd: torch.Tensor = None,
                time: torch.Tensor = None,
                enable_noise: bool = True) -> torch.Tensor:
        if state is None:
            state = self.state
        if last_state is None:
            last_state = self.last_state
        if cmd is None:
            cmd = self.cmd
        if time is None:
            time = self.time 
            
        if enable_noise and self.cfg.noise.obs_std > 0:
            noise = QuadrotorState(pos=torch.normal(
                mean=0., std=self.cfg.noise.obs_std, size=state.pos.shape))
            state = state + noise.to(self.device)
        
        all_obs = []

        self.update_history(state, last_state, cmd, time)

        all_obs.append(self.get_task_obs(state, time))
        if self.use_current_state:
            all_obs.append(self.get_state_obs(state))
        if self.obs_cfg.short_history_len > 0:
            all_obs.append(self.get_history_obs(self.obs_cfg.short_history_len, is_short=True))
        if self.obs_cfg.long_history_len > 0:
            all_obs.append(self.get_history_obs(self.obs_cfg.long_history_len))
        if self.int_obs_size > 0:
            all_obs.append(self.get_int_obs())
        if self.ext_obs_size > 0:
            all_obs.append(self.get_ext_obs(state, cmd))
            
        return torch.cat(all_obs, dim=-1)
    
    def update_history(self, state, last_state, cmd, time):
        if self.history_len <= 0:
            return
        
        if 'state' in self.obs_cfg.history_content:
            state_obs = self.get_state_obs(state)
            self.history_buffer['state'].update(state_obs)
        if 'action' in self.obs_cfg.history_content:
            self.history_buffer['action'].update(cmd)
        if 'ref_err' in self.obs_cfg.history_content:
            ref_err = self.get_ref_err(state, time)
            self.history_buffer['ref_err'].update(ref_err)
        if 'dyn_err' in self.obs_cfg.history_content:
            dyn_err = self.get_dyn_err(state, last_state, cmd)
            self.history_buffer['dyn_err'].update(dyn_err)
        
    def get_state_obs(self, 
                      state: QuadrotorState = None, 
                      body_frame: bool = None, 
                      use_rot_matrix: bool = None,
                      bodyrate: bool = None) -> torch.Tensor:
        if state is None:
            state = self.state
        if body_frame is None:
            body_frame = self.obs_cfg.use_body_frame
        if bodyrate is None:
            bodyrate = self.obs_cfg.bodyrate
        if use_rot_matrix is None:
            use_rot_matrix = self.obs_cfg.use_rot_matrix
            
        pos, rot, vel, ang_vel = state.pos, state.quat, state.vel, state.ang_vel
        if body_frame:
            pos = ru.inv_rotate_vector(pos, rot, 'quat')
            vel = ru.inv_rotate_vector(vel, rot, 'quat')
        if use_rot_matrix:
            rot = ru.quat2matrix(rot).reshape(-1, 9)
        if bodyrate:
            return torch.cat([pos, vel, rot, ang_vel], dim=-1)
        else:
            return torch.cat([pos, vel, rot], dim=-1)
    
    def get_dyn_err(self, 
                    state: QuadrotorState,
                    last_state: QuadrotorState,
                    cmd: torch.Tensor,) -> torch.Tensor:
        if self.use_rigid_body:
            last_state_vec = last_state.to_vec(self.obs_cfg.bodyrate)
            actual_state_vec = state.to_vec(self.obs_cfg.bodyrate)
            baseline_state_vec = self.predictive_model(last_state_vec, self.process_action(cmd.clone()))
        else:
            last_state_vec = self.get_state_obs(state, body_frame=False, bodyrate=True, use_rot_matrix=True)
            actual_state_vec = self.get_state_obs(state, body_frame=False, bodyrate=True, use_rot_matrix=False)
            baseline_state_vec = self.predictive_model.predict(last_state_vec, cmd, output_quat=True)

        if self.pred_model_noise > 0:
            noise = torch.normal(
                mean=torch.zeros_like(baseline_state_vec), 
                std=self.pred_model_noise)
            baseline_state_vec = baseline_state_vec + noise

        if self.obs_cfg.bodyrate:
            real_pos, real_vel, real_quat, real_angvel = actual_state_vec.split([3, 3, 4, 3], dim=-1)
            est_pos, est_vel, est_quat, est_angvel = baseline_state_vec.split([3, 3, 4, 3], dim=-1)
        else:
            real_pos, real_vel, real_quat = actual_state_vec.split([3, 3, 4], dim=-1)
            est_pos, est_vel, est_quat = baseline_state_vec.split([3, 3, 4], dim=-1)

        if self.obs_cfg.use_body_frame:
            real_pos = ru.inv_rotate_vector(real_pos, real_quat, 'quat')
            real_vel = ru.inv_rotate_vector(real_vel, real_quat, 'quat')
            est_pos = ru.inv_rotate_vector(est_pos, est_quat, 'quat')
            est_vel = ru.inv_rotate_vector(est_vel, est_quat, 'quat')

        obs = []
        if 'pos' in self.obs_cfg.dyn_err_content:
            pos_err = real_pos - est_pos
            obs.append(pos_err)
        if 'vel' in self.obs_cfg.dyn_err_content:
            vel_err = real_vel - est_vel
            obs.append(vel_err)
        if 'rot' in self.obs_cfg.dyn_err_content:
            quat_err = ru.cos_distance(real_quat, est_quat, 'quat', norm=False)
            obs.append(quat_err)
        if 'angvel' in self.obs_cfg.dyn_err_content:
            angvel_err = real_angvel - est_angvel
            obs.append(angvel_err)

        return torch.cat(obs, dim=-1)

    def get_history_obs(self, history_len: int, is_short: bool = False) -> torch.Tensor:
        history_obs = []
        for name, history in self.history_buffer.items():
            if 'err' in name:
                prefix = 'short' if is_short else 'long'
                include_diff = self.obs_cfg.get(f'{prefix}_diff', False)
                include_int = self.obs_cfg.get(f'{prefix}_int', False)
            else:
                include_diff = False
                include_int = False
            history_obs.append(history.tail(history_len, include_diff, include_int
                                            ).transpose(0, 1).flatten(1))
        return torch.cat(history_obs, dim=-1)
        
    def get_ext_obs(self,
                    state: QuadrotorState,
                    cmd: torch.Tensor,) -> torch.Tensor:
        """
        Return the current extrinsic observation.
        """
        return self.robots.get_obs_extrinsics(state, self.process_action(cmd.clone()))
        
    def get_int_obs(self,) -> torch.Tensor:
        """
        Return the current intrinsic observation.
        """
        return self.robots.get_obs_intrinsics(list(self.dynamics_randomization.keys()))

    @abc.abstractclassmethod
    def get_ref_err(self, state: QuadrotorState, time: torch.Tensor) -> torch.Tensor:
        """
        Return the current reference feedback at current time.
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_task_obs(self, state: QuadrotorState, time: torch.Tensor) -> torch.Tensor:
        """
        Return the current task-oriented observation.
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_task_reference(self, time: torch.Tensor = None) -> Dict:
        """
        Return the current task reference.
        """
        raise NotImplementedError
        
    def update_metrics(self, 
                       all_rewards: Dict=None, 
                       state: QuadrotorState=None, 
                       last_state: QuadrotorState=None, 
                       cmd: torch.Tensor=None, 
                       time: torch.Tensor=None
                       ):
        """
        Compute the metrics of the current episode.
        """
        # get metrics
        metrics = {}
        
        # update record
        if cmd is not None:
            self.record_max_thrust_cmd = torch.max(self.record_max_thrust_cmd, cmd[:, 0].abs())
            self.record_max_roll_rate_cmd = torch.max(self.record_max_roll_rate_cmd, cmd[:, 1].abs())
            self.record_max_pitch_rate_cmd = torch.max(self.record_max_pitch_rate_cmd, cmd[:, 2].abs())
            self.record_max_yaw_rate_cmd = torch.max(self.record_max_yaw_rate_cmd, cmd[:, 3].abs())
            
            metrics['max_thrust_cmd(m_s^2)'] = self.record_max_thrust_cmd
            metrics['max_roll_rate_cmd(deg_s)'] = self.record_max_roll_rate_cmd * 180. / torch.pi
            metrics['max_pitch_rate_cmd(deg_s)'] = self.record_max_pitch_rate_cmd * 180. / torch.pi
            metrics['max_yaw_rate_cmd(deg_s)'] = self.record_max_yaw_rate_cmd * 180. / torch.pi
            
        if state is not None:
            self.record_max_acc = torch.max(self.record_max_acc, state.acc.norm(dim=-1))
            self.record_max_vel = torch.max(self.record_max_vel, state.vel.norm(dim=-1))
            self.record_max_roll_rate = torch.max(self.record_max_roll_rate, state.ang_vel[:, 0].abs())
            self.record_max_pitch_rate = torch.max(self.record_max_pitch_rate, state.ang_vel[:, 1].abs())
            self.record_max_yaw_rate = torch.max(self.record_max_yaw_rate, state.ang_vel[:, 2].abs())
            self.record_max_pos_x = torch.max(self.record_max_pos_x, state.pos[:, 0].abs())
            self.record_max_pos_y = torch.max(self.record_max_pos_y, state.pos[:, 1].abs())
            self.record_max_pos_z = torch.max(self.record_max_pos_z, state.pos[:, 2].abs())
            self.record_max_roll = torch.max(self.record_max_roll, ru.quat2euler(state.quat)[:, 0].abs())
            self.record_max_pitch = torch.max(self.record_max_pitch, ru.quat2euler(state.quat)[:, 1].abs())
            self.record_max_yaw = torch.max(self.record_max_yaw, ru.quat2euler(state.quat)[:, 2].abs())
            self.record_avg_acc = self.record_avg_acc + state.acc.norm(dim=-1)
            self.record_avg_jerk = self.record_avg_jerk + (state.acc - last_state.acc).norm(dim=-1) / self.robots.control_dt
            self.record_avg_angvel = self.record_avg_angvel + state.ang_vel.norm(dim=-1)

            metrics['max_acc(m_s^2)'] = self.record_max_acc
            metrics['max_vel(m_s)'] = self.record_max_vel
            metrics['max_roll_rate(deg_s)'] = self.record_max_roll_rate * 180. / torch.pi
            metrics['max_pitch_rate(deg_s)'] = self.record_max_pitch_rate * 180. / torch.pi
            metrics['max_yaw_rate(deg_s)'] = self.record_max_yaw_rate * 180. / torch.pi
            metrics['max_pos_x(m)'] = self.record_max_pos_x
            metrics['max_pos_y(m)'] = self.record_max_pos_y
            metrics['max_pos_z(m)'] = self.record_max_pos_z
            metrics['max_roll(deg)'] = self.record_max_roll * 180. / torch.pi
            metrics['max_pitch(deg)'] = self.record_max_pitch * 180. / torch.pi
            metrics['max_yaw(deg)'] = self.record_max_yaw * 180. / torch.pi
            metrics['avg_acc(m_s^2)'] = self.record_avg_acc / self.progress.float()
            metrics['avg_jerk(m_s^3)'] = self.record_avg_jerk / self.progress.float()
            metrics['avg_angvel(deg_s)'] = self.record_avg_angvel / self.progress.float()

            if 'dyn_err' in self.obs_cfg.history_content:
                self.record_pred_dyn_err = self.record_pred_dyn_err + self.history_buffer['dyn_err'].tail(1).squeeze(0).abs()
                metrics['pred_dyn_err'] = self.record_pred_dyn_err.mean(-1) / self.progress.float()
                for i in range(self.dyn_err_size):
                    metrics[f'pred_dyn_err/{i}'] = self.record_pred_dyn_err[:, i] / self.progress.float()
             
        if self.ext_obs_size > 0:
            self.record_error_l1ac += torch.norm(self.robots.wind_vec - self.robots.l1ac_wind_vec, dim=-1)
            metrics['error_l1ac(m_s^2)'] = self.record_error_l1ac / self.progress.float()
        
        if all_rewards is not None:
            for key, value in all_rewards.items():
                if hasattr(self, f'record_{key}'):
                    setattr(self, f'record_{key}', getattr(self, f'record_{key}') + value)
                else:
                    self.eval_metrics.append(key)
                    setattr(self, f'record_{key}', value)
                metrics[key] = getattr(self, f'record_{key}')

        
        return metrics

    ########################## Env Specs ##########################
    def setup_specs(self):
        # observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        # action space
        if self.robots.control_mode == 'ctbr':
            action_range = []
            if isinstance(self.action_cfg.thrust, (int, float)):
                scale = self.action_cfg.thrust
                thrust_range = [self.robots.g * max(2. - scale, 0.), self.robots.g * min(scale * 1., 4.)]
            else:
                thrust_range = list(self.action_cfg.thrust)
            action_range.append(thrust_range)
            for omega in ['roll_rate', 'pitch_rate', 'yaw_rate']:
                scale = self.action_cfg.get(f'{omega}')
                if isinstance(scale, (int, float)):
                    omega_range = [-np.pi * scale, np.pi * scale]
                else:
                    omega_range = np.array(scale) * np.pi / 180.
                action_range.append(omega_range)
            action_range = np.asarray(action_range)
            self.action_range = torch.from_numpy(action_range).float().to(self.device)
            if self.action_cfg.offset_g:
                self.action_range[0] -= self.robots.g
        elif self.robots.control_mode == 'srt':
            if isinstance(self.action_cfg.thrust, (int, float)):
                scale = self.action_cfg.thrust
                thrust_range = [self.robots.g * max(2. - scale, 0.), self.robots.g * min(scale * 1., 4.)]
            else:
                thrust_range = list(self.action_cfg.thrust)
            action_range = np.asarray([thrust_range] * 4) / 4. # shape: (4, 2)
            self.action_range = torch.from_numpy(action_range).float().to(self.device)
            if self.action_cfg.offset_g:
                self.action_range[:] -= self.robots.g / 4
        else:
            raise NotImplementedError(f'Invalid control mode: {self.robots.control_mode}')

        self.action_space = spaces.Box(
            low=self.action_range[..., 0].cpu().numpy(), 
            high=self.action_range[..., 1].cpu().numpy(),
            shape=(self.action_size,), dtype=np.float32)
        print("[Env.action_space]", self.action_space.low, self.action_space.high)

    def init_vars(self):    
        # initialize the environment state
        self.state = self.robots.state.clone()
        self.last_state = self.robots.state.clone()
        self.cmd = torch.zeros(self.num_envs, self.action_size, dtype=torch.float32, device=self.device)
        self.last_cmd = self.cmd.clone()
        self.time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.acc_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.progress = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        self.history_len = max(self.obs_cfg.long_history_len, self.obs_cfg.short_history_len)
        self.use_current_state = False if self.history_len > 0 else True

        self.boundaries = self.state.pos.unsqueeze(-1) + self.boundary_offset
        
        if self.cfg.latency_obs > 0:
            buffer_len = ceil(self.cfg.latency_obs / self.robots.control_dt)
            self.obs_buffer = HistoryQueue(buffer_len, self.num_envs, self.obs_size, device=self.device)
        if self.cfg.latency_cmd > 0:
            buffer_len = ceil(self.cfg.latency_cmd / self.robots.control_dt)
            self.action_buffer = HistoryQueue(buffer_len, self.num_envs, self.action_size, device=self.device)

        self.eval_metrics = ['max_thrust_cmd(m_s^2)', 
                             'max_roll_rate_cmd(deg_s)', 
                             'max_pitch_rate_cmd(deg_s)', 
                             'max_yaw_rate_cmd(deg_s)',
                             'max_pos_x(m)',
                             'max_pos_y(m)',
                             'max_pos_z(m)',
                             'max_roll(deg)',
                             'max_pitch(deg)',
                             'max_yaw(deg)',
                             'max_acc(m_s^2)', 
                             'max_vel(m_s)', 
                             'max_roll_rate(deg_s)', 
                             'max_pitch_rate(deg_s)', 
                             'max_yaw_rate(deg_s)',
                             'avg_acc(m_s^2)',
                             'avg_jerk(m_s^3)',
                             'avg_angvel(deg_s)',]
        
        if self.ext_obs_size > 0:
            self.eval_metrics.append('error_l1ac(m_s^2)')
        
        for m in self.eval_metrics:
            setattr(self, f'record_{m.split("(")[0]}', torch.zeros(self.num_envs, dtype=torch.float32, device=self.device))

        if 'dyn_err' in self.obs_cfg.history_content:
            self.eval_metrics.append('pred_dyn_err')
            self.record_pred_dyn_err = torch.zeros((self.num_envs, self.dyn_err_size), dtype=torch.float32, device=self.device)

    def init_history(self):
        if self.history_len > 0:
            self.history_buffer = {}
            for term in self.obs_cfg.history_content:
                data_size = getattr(self, f'{term}_size')
                self.history_buffer[term] = HistoryQueue(
                    self.history_len + 10, self.num_envs, data_size, device=self.device)
            state_obs = self.get_state_obs(self.state)
            self.history_buffer['state'].reset(init_data=state_obs)

    def setup_randomizations(self):
        self.state_randomization = {}
        self.dynamics_randomization = {}

        sections = ['state', 'dynamics']
        cfgs = [self.task_cfg.state_randomization, self.quadrotor_cfg.dynamics_randomization]
        for sec, cfg in zip(sections, cfgs):
            if sec == 'state' and not self.random_init:
                continue

            if sec == 'dynamics' and not self.random_dynamics:
                continue

            for item, params in cfg.items():
                scale = params.get('scale', None)
                v_min = params.get('min', None)
                v_max = params.get('max', None)
                
                if scale is not None:
                    if isinstance(scale, (int, float)):
                        scale = [1.-float(scale), 1.+float(scale)]
                    else:
                        assert isinstance(scale, (ListConfig, list)) and len(scale) == 2, \
                            f"Invalid scale for {sec} {item}: {scale}"
                        
                    base_value = to_tensor(getattr(self.robots, '_' + item), self.device)
                    
                    v_min_tensor = base_value * float(scale[0])
                    v_max_tensor = base_value * float(scale[1])
                else:
                    # Convert to tensors if not None
                    v_min_tensor = to_tensor(v_min, self.device)
                    v_max_tensor = to_tensor(v_max, self.device)

                # Assign the negative value if only one is provided
                if v_min_tensor is not None and v_max_tensor is None:
                    v_max_tensor = -1. * v_min_tensor
                elif v_min_tensor is None and v_max_tensor is not None:
                    v_min_tensor = -1. * v_max_tensor

                # Create the Uniform distribution if both values are present
                if v_min_tensor is not None and v_max_tensor is not None:
                    try:
                        getattr(self, f'{sec}_randomization')[item] = D.Uniform(v_min_tensor, v_max_tensor)
                    except:
                        print(f"[WARNING] Invalid randomization for {sec} {item}: {v_min_tensor} ~ {v_max_tensor}")
                    
        print("[Info] Finished setting up randomizations:")
        print(">>> State randomization:")
        for key, value in self.state_randomization.items():
            print(f"    {key}: {value.low} ~ {value.high}")
        print(">>> Dynamics randomization:")
        for key, value in self.dynamics_randomization.items():
            print(f"    {key}: {value.low} ~ {value.high}")
        print("--")

    ########################## Env Properties ##########################
    @property
    def state_size(self):
        dim = 3 + 3 # pos + vel
        if self.obs_cfg.use_rot_matrix:
            dim += 9 # rot_matrix
        else:
            dim += 4 # quat
        if self.obs_cfg.bodyrate:
            dim += 3
        return dim
    
    @property
    def action_size(self):
        return 4
    
    @property
    def ref_err_size(self):
        return 3
    
    @property
    def dyn_err_size(self):
        dim = 3 * len(self.obs_cfg.dyn_err_content)
        return dim
        
    @property
    def short_history_obs_size(self):
        """
        Return the size of the trajectory feedback observation.
        """
        if self.history_len > 0:
            obs_dim = 0
            if 'state' in self.obs_cfg.history_content:
                obs_dim += self.state_size
            if 'action' in self.obs_cfg.history_content:
                obs_dim += self.action_size
            if 'ref_err' in self.obs_cfg.history_content:
                obs_dim += self.ref_err_size * (1 + int(self.obs_cfg.get('short_diff', False)) + int(self.obs_cfg.get('short_int', False)))
            if 'dyn_err' in self.obs_cfg.history_content:
                obs_dim += self.dyn_err_size * (1 + int(self.obs_cfg.get('short_diff', False)) + int(self.obs_cfg.get('short_int', False)))
            return obs_dim
        else:
            return 0

    @property
    def long_history_obs_size(self):
        """
        Return the size of the trajectory feedback observation.
        """
        if self.history_len > 0:
            obs_dim = 0
            if 'state' in self.obs_cfg.history_content:
                obs_dim += self.state_size
            if 'action' in self.obs_cfg.history_content:
                obs_dim += self.action_size
            if 'ref_err' in self.obs_cfg.history_content:
                obs_dim += self.ref_err_size * (1 + int(self.obs_cfg.get('long_diff', False)) + int(self.obs_cfg.get('long_int', False)))
            if 'dyn_err' in self.obs_cfg.history_content:
                obs_dim += self.dyn_err_size * (1 + int(self.obs_cfg.get('long_diff', False)) + int(self.obs_cfg.get('long_int', False)))
            return obs_dim
        else:
            return 0
        
    @property
    def ext_obs_size(self):
        """
        Return the size of the extrinsic observation.
        """
        if not self.obs_cfg.extrinsics:
            return 0
        else:
            return 3
        
    @property
    def int_obs_size(self):
        """
        Return the size of the intrinsic observation.
        """
        if not self.obs_cfg.intrinsics or len(self.dynamics_randomization) == 0:
            return 0
        else:
            return self.get_int_obs().shape[1]
    
    @abc.abstractproperty
    def task_obs_size(self):
        """
        Return the size of the task observation.
        """
        raise NotImplementedError

    @property
    def obs_size(self):
        obs_dim = np.prod(self.task_obs_size, dtype=np.int32) \
            + self.obs_cfg.short_history_len * self.short_history_obs_size \
            + self.obs_cfg.long_history_len * self.long_history_obs_size \
            + self.ext_obs_size + self.int_obs_size 
        if self.use_current_state:
            obs_dim += self.state_size
        return int(obs_dim)
            
    @property
    def obs_size_dict(self):
        return {
            'task_obs': self.task_obs_size,
            'state_obs': self.state_size if self.use_current_state else 0,
            'short_history_obs': (self.obs_cfg.short_history_len, self.short_history_obs_size),
            'long_history_obs': (self.obs_cfg.long_history_len, self.long_history_obs_size),
            'int_obs': self.int_obs_size,
            'ext_obs': self.ext_obs_size,
        }

def register_task(name, cls):
    if name in GymEnv.REGISTRY:
        raise ValueError(f"Cannot register duplicate task ({name})")
    else:
        print(f"[Info] Registering task: {name}")
    GymEnv.REGISTRY[name] = cls
    GymEnv.REGISTRY[name.lower()] = cls
    
def get_task(name):
    if name not in GymEnv.REGISTRY:
        raise ValueError(f"Cannot find controller ({name})")
    return GymEnv.REGISTRY[name]
