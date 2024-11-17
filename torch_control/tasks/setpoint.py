from typing import Dict

import gym.spaces as spaces
import numpy as np
import torch
import torch.distributions as D
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.tasks.base import GymEnv, register_task
from torch_control.utils import rot_utils as ru


class SetPoint(GymEnv):

    def __init__(self, cfg, num_envs: int):
        super().__init__(cfg, num_envs)

        self.target_rel_pos_dist = D.Uniform(
            torch.tensor([-3.0, -3.0, -3.0], dtype=torch.float32, device=self.device),
            torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32, device=self.device)
        )
        self.target_yaw_dist = D.Uniform(
            torch.tensor(-np.pi, dtype=torch.float32, device=self.device),
            torch.tensor(np.pi, dtype=torch.float32, device=self.device)
        )

        self.eval_metrics += ['error_x(m)', 'error_y(m)', 'error_z(m)', 'error_distance(m)',
                              'error_roll(deg)', 'error_pitch(deg)', 'error_yaw(deg)']

    @property
    def task_obs_size(self):
        if self.obs_cfg.use_rot_matrix:
            return 3 + 9
        else:
            return 3 + 4 # pos_to_target + quat_to_target
    
    @torch.no_grad()
    def reset_task(self, idx: torch.Tensor = None, task_vec: torch.Tensor = None):
        if idx is None:
            idx = torch.arange(self.num_envs)
        
        if task_vec is not None:
            pos_vec, quat_vec = torch.split(task_vec, [3, 4], dim=-1)
            
        robot_pos = self.robots.state.pos
        
        if not hasattr(self, 'target_pos'):
            self.target_pos = robot_pos + self.target_rel_pos_dist.sample((self.num_envs,))

        if task_vec is not None:
            self.target_pos[idx] = pos_vec[idx] # robot_pos[idx] + 
        else:
            self.target_pos[idx] = robot_pos[idx] + self.target_rel_pos_dist.sample((len(idx),))

        if not hasattr(self, 'target_yaw'):
            self.target_yaw = self.target_yaw_dist.sample((self.num_envs,))
            self.target_ang = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
            self.target_ang[..., 2] = self.target_yaw

        if task_vec is not None:
            self.target_ang[idx] = ru.quat2euler(quat_vec[idx])
            self.target_quat = ru.euler2quat(self.target_ang)
            print("===== Forcibly reset task target =====")
            print("[SetPoint] target_pos (meter): ", self.target_pos[idx])
            print("[SetPoint] target_ang (degree): ", self.target_ang[idx] * 180. / torch.pi)
            print("======================================")
        else:
            self.target_yaw[idx] = self.target_yaw_dist.sample((len(idx),))
            self.target_ang[idx, 2] = self.target_yaw[idx]
            self.target_quat = ru.euler2quat(self.target_ang)
        
    def get_task_reference(self, time: torch.Tensor = None) -> Dict:
        return {
            'position': self.target_pos,
            'orientation': self.target_quat,
            'yaw': self.target_yaw
        }
        
    def get_task_obs(self, state: QuadrotorState, time: torch.Tensor = None) -> torch.Tensor:
        if state is None:
            state = self.state
            
        pos_to_target = self.target_pos - state.pos
        rot_to_target = ru.delta_quat(state.quat, self.target_quat)
        if self.obs_cfg.use_rot_matrix:
            rot_to_target = ru.quat2matrix(rot_to_target).reshape(self.num_envs, 9)
        task_obs = torch.cat([pos_to_target, rot_to_target], dim=-1)
        return task_obs
    
    def get_ref_err(self, state: QuadrotorState, time: torch.Tensor = None) -> torch.Tensor:
        if state is None:
            state = self.state
            
        return self.target_pos - state.pos

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
        
        pos_error = torch.norm(self.target_pos - state.pos, dim=-1)
        reward_pos = torch.exp(-pos_error)
        
        rot_error = ru.euler_distance(self.target_quat, state.quat)
        reward_rot = torch.exp(-rot_error)
        
        rewards = {
            'reward_pos': 1. * reward_pos,
            'reward_rot': 0.5 * reward_rot,
        }
        base_rewards.update(rewards)
        
        return base_rewards
    
    def update_metrics(self,
                       all_rewards: Dict=None, 
                       state: QuadrotorState=None, 
                       last_state: QuadrotorState=None, 
                       cmd: torch.Tensor=None, 
                       time: torch.Tensor=None):
        metrics = super().update_metrics(all_rewards, state, last_state, cmd, time)
        
        if state is not None:
            metrics['error_x(m)'] = (self.target_pos[..., 0] - state.pos[..., 0]).abs()
            metrics['error_y(m)'] = (self.target_pos[..., 1] - state.pos[..., 1]).abs()
            metrics['error_z(m)'] = (self.target_pos[..., 2] - state.pos[..., 2]).abs()
            metrics['error_distance(m)'] = torch.norm(self.target_pos - state.pos, dim=-1)
            
            ang = ru.quat2euler(state.quat)
            metrics['error_roll(deg)'] = (self.target_ang[..., 0] - ang[..., 0]).abs() * 180. / torch.pi
            metrics['error_pitch(deg)'] = (self.target_ang[..., 1] - ang[..., 1]).abs() * 180. / torch.pi
            metrics['error_yaw(deg)'] = (self.target_ang[..., 2] - ang[..., 2]).abs() * 180. / torch.pi
        
        return metrics
        
register_task('setpoint', SetPoint)
