from typing import Dict

import gym.spaces as spaces
import numpy as np
import torch
import torch.distributions as D
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.tasks.base import GymEnv, register_task
from torch_control.tasks.trajectory import get_trajectory
from torch_control.utils import rot_utils as ru


class Track(GymEnv):

    def __init__(self, cfg, num_envs: int):
        self.traj_type = cfg.task.trajectory.type
        self.traj_future_steps = int(cfg.task.trajectory.future_steps)
        self.traj_step_size = int(cfg.task.trajectory.step_size)
        if cfg.task.trajectory.get('origin', None) is not None:
            self.traj_origin = torch.tensor(list(cfg.task.trajectory.origin)).float()
        else:
            self.traj_origin = None
        self.distance_threshold = cfg.task.trajectory.distance_threshold
        self.warmup_steps = cfg.task.warmup_steps
        
        super().__init__(cfg, num_envs)

        self.ref = get_trajectory(self.traj_type, 
                                  num_trajs=num_envs, 
                                  origin=self.traj_origin,
                                  **self.task_cfg.trajectory[self.traj_type],
                                  device=self.device,
                                  seed=cfg.seed)
        
        self.cum_error_x = torch.zeros(num_envs).to(self.device)
        self.cum_error_y = torch.zeros(num_envs).to(self.device)
        self.cum_error_z = torch.zeros(num_envs).to(self.device)
        self.cum_error_vx = torch.zeros(num_envs).to(self.device)
        self.cum_error_vy = torch.zeros(num_envs).to(self.device)
        self.cum_error_vz = torch.zeros(num_envs).to(self.device)
        self.cum_error_roll = torch.zeros(num_envs).to(self.device)
        self.cum_error_pitch = torch.zeros(num_envs).to(self.device)
        self.cum_error_yaw = torch.zeros(num_envs).to(self.device)
        self.cum_error_distance = torch.zeros(num_envs).to(self.device)

        self.eval_metrics += ['error_x(m)', 'error_y(m)', 'error_z(m)', 'error_distance(m)',
                              'error_vx(m_s)', 'error_vy(m_s)', 'error_vz(m_s)',
                              'error_roll(deg)', 'error_pitch(deg)', 'error_yaw(deg)']

    @property
    def task_obs_size(self):
        return (self.traj_future_steps, 3)
    
    @torch.no_grad()
    def reset_task(self, idx: torch.Tensor = None, task_vec: torch.Tensor = None):
        if idx is None:
            idx = torch.arange(self.num_envs)
            
        robot_pos = self.robots.state.pos
            
        if task_vec is not None:
            rel_pos_vec, _ = torch.split(task_vec, [3, 4], dim=-1)
            pos_vec = robot_pos + rel_pos_vec
            self.ref.reset(idx, pos_vec[idx], verbose=True)
            print("===== Forcibly reset task target =====")
            print("[Track] trajectory origin (meter): ", self.ref.origin)
            print("======================================")
        elif self.elapsed_steps == 0 or self.elapsed_steps > self.warmup_steps:
            self.ref.reset(idx, robot_pos[idx])
        
        self.cum_error_x[idx] = 0.
        self.cum_error_y[idx] = 0.
        self.cum_error_z[idx] = 0.
        self.cum_error_vx[idx] = 0.
        self.cum_error_vy[idx] = 0.
        self.cum_error_vz[idx] = 0.
        self.cum_error_roll[idx] = 0.
        self.cum_error_pitch[idx] = 0.
        self.cum_error_yaw[idx] = 0.
        self.cum_error_distance[idx] = 0.

    def get_task_reference(self, time: torch.Tensor = None) -> Dict:
        if time is None:
            time = self.time
        ref_pos = self.ref.pos(time).to(self.device)
        ref_vel = self.ref.vel(time).to(self.device)
        ref_acc = self.ref.acc(time).to(self.device)
        ref_yaw = self.ref.yaw(time).to(self.device)
        ref_yaw_vel = self.ref.yawvel(time).to(self.device)
        # ref_ang = torch.zeros(self.num_envs, 3).to(self.device)
        ref_ang = torch.zeros_like(ref_pos)
        ref_ang[..., 2] = ref_yaw
        ref_quat = ru.euler2quat(ref_ang)
        return {
            'position': ref_pos,
            'orientation': ref_quat,
            'linear_velocity': ref_vel,
            'acceleration': ref_acc,
            'heading_rate': ref_yaw_vel
        }

    def get_task_obs(self, state: QuadrotorState, time: torch.Tensor = None) -> torch.Tensor:
        if state is None:
            state = self.state
        if time is None:
            time = self.time

        future_ref = [self.ref.pos(time + self.traj_step_size * i * self.robots.control_dt
                                   ).to(self.device) for i in range(self.traj_future_steps)]
        
        if self.obs_cfg.use_body_frame:
            pos = ru.inv_rotate_vector(state.pos, state.quat, 'quat')
            ff_term = [pos - ru.inv_rotate_vector(ff, state.quat, 'quat') 
                       for ff in future_ref]
        else:
            ff_term = [state.pos - ff for ff in future_ref]
        
        task_obs = torch.cat(ff_term, dim=-1)
        
        return task_obs
    
    def get_ref_err(self, state: QuadrotorState, time: torch.Tensor) -> torch.Tensor:
        if state is None:
            state = self.state
        if time is None:
            time = self.time
            
        current_ref = self.ref.pos(time).to(self.device)
        if self.obs_cfg.use_body_frame:
            pos = ru.inv_rotate_vector(state.pos, state.quat, 'quat')
            target_pos = ru.inv_rotate_vector(current_ref, state.quat, 'quat')
            return pos - target_pos
        else:
            return state.pos - current_ref

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
        
        base_rewards = super().get_reward(state, last_state, cmd, last_cmd, time)
        
        pos_error = torch.norm(
            self.ref.pos(self.time).to(self.device) - state.pos, dim=-1)
        reward_pos = torch.exp(-pos_error)
        
        yaw_error = torch.abs(
            self.ref.yaw(self.time).to(self.device) - ru.quat2euler(state.quat)[..., 2])
        reward_yaw = torch.exp(-yaw_error)
        
        vel_error = torch.norm(
            self.ref.vel(self.time).to(self.device) - state.vel, dim=-1)
        reward_vel = torch.exp(-vel_error)
        
        rewards = {
            'reward_pos': 1. * reward_pos,
            'reward_rot': 0.2 * reward_yaw,
            'reward_vel': 0.1 * reward_vel,
        }
        base_rewards.update(rewards)
        
        return base_rewards
    
    def check_safety_constraints(self, state: QuadrotorState = None):
        if state is None:
            state = self.state
        
        is_safe = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        is_safe &= torch.norm(
            self.ref.pos(self.time).to(self.device) - state.pos, dim=-1) < self.distance_threshold
        
        return is_safe
    
    @torch.no_grad()
    def update_metrics(self, 
                       all_rewards: Dict=None, 
                       state: QuadrotorState=None, 
                       last_state: QuadrotorState=None, 
                       cmd: torch.Tensor=None, 
                       time: torch.Tensor=None):
        metrics = super().update_metrics(all_rewards, state, last_state, cmd, time)
            
        if state is not None:
            assert time is not None, "time must be provided to compute metrics"
            
            ref_pos = self.ref.pos(time).to(self.device)
            ref_vel = self.ref.vel(time).to(self.device)
            ref_yaw = self.ref.yaw(time).to(self.device)
                
            ang = ru.quat2euler(state.quat)
            
            self.cum_error_x += (ref_pos[..., 0] - state.pos[..., 0]).abs()
            self.cum_error_y += (ref_pos[..., 1] - state.pos[..., 1]).abs()
            self.cum_error_z += (ref_pos[..., 2] - state.pos[..., 2]).abs()
            self.cum_error_vx += (ref_vel[..., 0] - state.vel[..., 0]).abs()
            self.cum_error_vy += (ref_vel[..., 1] - state.vel[..., 1]).abs()
            self.cum_error_vz += (ref_vel[..., 2] - state.vel[..., 2]).abs()
            self.cum_error_roll += (0. - ang[..., 0]).abs() * 180. / torch.pi
            self.cum_error_pitch += (0. - ang[..., 1]).abs() * 180. / torch.pi
            self.cum_error_yaw += (ref_yaw - ang[..., 2]).abs() * 180. / torch.pi
            self.cum_error_distance += torch.norm(ref_pos - state.pos, dim=-1)
            
            metrics['error_distance(m)'] = self.cum_error_distance / self.progress.float()
            metrics['error_x(m)'] = self.cum_error_x / self.progress.float()
            metrics['error_y(m)'] = self.cum_error_y / self.progress.float()
            metrics['error_z(m)'] = self.cum_error_z / self.progress.float()
            metrics['error_vx(m_s)'] = self.cum_error_vx / self.progress.float()
            metrics['error_vy(m_s)'] = self.cum_error_vy / self.progress.float()
            metrics['error_vz(m_s)'] = self.cum_error_vz / self.progress.float()
            metrics['error_roll(deg)'] = self.cum_error_roll / self.progress.float()
            metrics['error_pitch(deg)'] = self.cum_error_pitch / self.progress.float()
            metrics['error_yaw(deg)'] = self.cum_error_yaw / self.progress.float()
        
        return metrics
        
register_task('track', Track)