import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

from torch_control.controllers.base import Controller, register_controller
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.tasks.base import GymEnv
from torch_control.utils import rot_utils as ru


class PID(Controller):
    
    def __init__(self, cfg, envs: GymEnv, eval_envs_dict: Dict[str, GymEnv], device: str):
        super().__init__(cfg, envs, eval_envs_dict, device)
        self.l1ac_log = [torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])]
        self.prev_t = None
        self.dt = None
        self.g = 9.81
        
    def setup_params(self, ctrl_freq: float = None):
        self.kp_pos_xy = self.cfg.kp_pos_xy
        self.kp_pos_z = self.cfg.kp_pos_z
        self.kd_pos_xy = self.cfg.kd_pos_xy
        self.kd_pos_z = self.cfg.kd_pos_z
        self.ki_pos_xy = self.cfg.ki_pos_xy
        self.ki_pos_z = self.cfg.ki_pos_z
        self.int_limit_xy = self.cfg.int_limit_xy
        self.int_limit_z = self.cfg.int_limit_z
        self.kp_rot = self.cfg.kp_rot
        self.kp_yaw = self.cfg.kp_yaw
        self.kp_rot_rp = self.cfg.kp_rot_rp
        print(f"kp_pos_xy: {self.kp_pos_xy}, kp_pos_z: {self.kp_pos_z}, kd_pos_xy: {self.kd_pos_xy}, kd_pos_z: {self.kd_pos_z}, ki_pos_xy: {self.ki_pos_xy}, ki_pos_z: {self.ki_pos_z}, int_limit_xy: {self.int_limit_xy}, int_limit_z: {self.int_limit_z}, kp_rot: {self.kp_rot}, kp_yaw: {self.kp_yaw}")
        
        if ctrl_freq is None:
            self.ctrl_freq = self.envs.robots.control_freq
        else:
            self.ctrl_freq = ctrl_freq
        self.ctrl_dt = 1. / self.ctrl_freq
        
        self.pos_err_int = None
        self.count = 0
        self.v_prev = None

    def init_aux_vars(self, num_envs):
        self.pos_err_int = None
        self.prev_t = None
        self.dt = self.ctrl_dt
        self.count = 0
        self.v_prev = None
        
    def __call__(self, pid_obs: Dict[str, Any]) -> Tuple[torch.Tensor]:
        state = pid_obs.get('state') # QuadrotorState
        reference = pid_obs.get('reference') # Dict
        time = pid_obs.get('time') # torch.Tensor

        pos, vel, quat = state.pos, state.vel, state.quat
        euler_est = ru.quat2euler(quat)

        z_vec = torch.tensor([0, 0, 1]).float().to(self.device)
        z_vec = z_vec.unsqueeze(0).expand_as(pos)

        ref_euler = ru.quat2euler(reference.get('orientation', 
                                                torch.tensor([1.0, 0.0, 0.0, 0.0]).to(quat.device).expand_as(quat)))
        roll_ref, pitch_ref, yaw_ref = ref_euler[..., 0], ref_euler[..., 1], ref_euler[..., 2]
        
        pos_err = pos - reference.get('position', torch.zeros_like(pos))
        vel_err = vel - reference.get('linear_velocity', torch.zeros_like(vel))
        yaw_err = euler_est[..., 2] - yaw_ref
        if torch.any(yaw_err >=  torch.pi):
            yaw_err_wraped = torch.where(yaw_err >= torch.pi, yaw_err - 2.0 * torch.pi, yaw_err)
        elif torch.any(yaw_err < -torch.pi):
            yaw_err_wraped = torch.where(yaw_err < -torch.pi, yaw_err + 2.0 * torch.pi, yaw_err)
        else:
            yaw_err_wraped = torch.where((yaw_err >= -torch.pi) & (yaw_err < torch.pi), yaw_err, yaw_err)
        
        if self.pos_err_int is None:
            self.pos_err_int = torch.zeros_like(pos)
        self.pos_err_int = self.pos_err_int + pos_err * self.dt
        self.pos_err_int[..., :2] = torch.clamp(self.pos_err_int[..., :2], -self.int_limit_xy, self.int_limit_xy)
        self.pos_err_int[...,  2] = torch.clamp(self.pos_err_int[...,  2], -self.int_limit_z,  self.int_limit_z )
        
        g_vec = torch.tensor([0, 0, self.g]).float().to(self.device)
        g_vec = g_vec.unsqueeze(0).expand_as(pos)
        
        acc_des = g_vec \
            + reference.get('acceleration', torch.zeros_like(pos))
        acc_des[..., :2] = acc_des[..., :2] \
            -self.kp_pos_xy * pos_err[..., :2] \
            -self.kd_pos_xy * vel_err[..., :2] \
            -self.ki_pos_xy * self.pos_err_int[..., :2]
        acc_des[..., 2] = acc_des[..., 2] \
            -self.kp_pos_z * pos_err[..., 2] \
            -self.kd_pos_z * vel_err[..., 2] \
            -self.ki_pos_z * self.pos_err_int[..., 2]
            
        u_des = ru.inv_rotate_vector(acc_des, quat, mode='quat')
        acc_des = torch.norm(u_des, dim=-1, keepdim=True)
        rot_err = torch.linalg.cross(u_des / acc_des, z_vec)
        omega_des = -1. * self.kp_rot * rot_err
        
        yaw_feedback_des = torch.zeros_like(omega_des)
        yaw_feedback_des[..., 2] = self.kp_yaw * (yaw_err_wraped - reference.get('heading_rate', torch.zeros_like(omega_des[..., 2])))
        omega_des[..., 2] = omega_des[..., 2] - self.kp_yaw * yaw_err_wraped
    
        
        self.count += 1
        self.v_prev = vel
        
        return torch.cat([acc_des, omega_des], dim=-1)

    def update_dt(self, t):
        if self.prev_t is None:
            dt = self.ctrl_dt
        else:
            dt = t - self.prev_t
        
        if dt < 0.001 or dt > 0.1:
            dt = 0.02

        if self.dt is not None:
            self.dt = self.dt * 0.8 + dt * 0.2
        else:
            self.dt = dt
            
        self.prev_t = t
    
    def export_deploy_funcs(self):
        self.set_eval()
        self.phase = 'deploy'
        self.time = None
        
        def init_task(odom_dict: Dict, task_dict: Dict = None):
            state_vec = torch.cat([odom_dict['position'], 
                                   odom_dict['linear_velocity'],
                                   odom_dict['orientation'],
                                   odom_dict['angular_velocity']], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            
            if task_dict is not None:
                task_vec = torch.cat([task_dict['position'],
                                      task_dict['orientation']], dim=-1)
                task_vec = task_vec.unsqueeze(0).float().to(self.device)
            else:
                task_vec = None
            
            self.envs.reset(state_vec=state_vec, task_vec=task_vec)
            for eval_name in self.eval_envs_dict:
                self.eval_envs_dict[eval_name].reset(state_vec=state_vec, task_vec=task_vec)
            
            self.last_state = self.envs.state.clone()
            self.last_cmd = self.envs.cmd.clone()

        def infer(odom_dict: Dict, time: float = None, last_cmd: torch.Tensor = None) -> torch.Tensor:
            if last_cmd is None:
                last_cmd = self.last_cmd
            last_cmd = last_cmd.to(self.device)
            
            state_vec = torch.cat([odom_dict['position'], 
                                   odom_dict['linear_velocity'],
                                   odom_dict['orientation'],
                                   odom_dict['angular_velocity']], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            state = QuadrotorState.construct_from_vec(state_vec).to(self.device)

            time_vec = torch.tensor([time]).float().to(self.device)

            ref = self.envs.get_task_reference(time_vec)
            self.envs.progress += 1
            self.last_state = state.clone()

            action = self.__call__({'state': state,
                                    'reference': ref,
                                    'time': time_vec})
            self.last_cmd = action.clone()

            # add logs
            logging_data = {
                'time': time_vec.squeeze(0).tolist(),
                'pos_est': state.pos.squeeze(0).tolist(),
                'pos_ref': ref['position'].squeeze(0).tolist(),
                'vel_est': state.vel.squeeze(0).tolist(),
                'vel_ref': ref['linear_velocity'].squeeze(0).tolist(),
                'att_est': state.quat.squeeze(0).tolist(),
                'att_ref': ref['orientation'].squeeze(0).tolist(),
                'omg_est': state.ang_vel.squeeze(0).tolist(),
                'omg_ref': action.squeeze(0)[1:].tolist(),
                'thr_ref': action.squeeze(0)[:1].tolist(),
            }

            logging_info = {
                'time': time,
                'state': odom_dict,
                'ref': {k: v.squeeze(0) for k, v in ref.items()},
                'l1ac': self.l1ac_log
            }

            return action.squeeze(0), logging_data, logging_info
            
        def preprocess(odom_dict: Dict, time: float = None) -> torch.Tensor:
            state_vec = torch.cat([odom_dict['position'], 
                                   odom_dict['linear_velocity'],
                                   odom_dict['orientation'],
                                   odom_dict['angular_velocity']], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            state = QuadrotorState.construct_from_vec(state_vec).to(self.device)
            
            time_vec = torch.tensor([time]).float().to(self.device)

            ref = self.envs.get_task_reference(time_vec)
            self.envs.progress += 1
            self.last_state = state.clone()
            
            logging_dict = {
                'time': time,
                'state': odom_dict,
                'ref': {k: v.squeeze(0) for k, v in ref.items()},
                'l1ac': self.l1ac_log
            }
            
            return {'state': state, 'reference': ref, 'time': time_vec}, logging_dict
        
        def controller(obs: Dict) -> torch.Tensor:
            return self.__call__(obs)
        
        def postprocess(actions: torch.Tensor) -> torch.Tensor:
            self.last_cmd = actions.clone()
            return actions.squeeze(0)
        
        return init_task, infer
    
    def save_params(self, name):
        pass
    
    def _tune_params(self):
        pass      
    
    def set_train(self):
        pass
    
    def set_eval(self):
        pass
    
register_controller('pid', PID)