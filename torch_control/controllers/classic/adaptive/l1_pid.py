import os
from typing import Any, Dict, Tuple

import torch

from torch_control.controllers.classic.basic.pid import PID, register_controller
from torch_control.controllers.l1ac import L1AC_batch
from torch_control.tasks.base import GymEnv
from torch_control.utils import rot_utils as ru


class L1_PID(PID):
    def __init__(self, cfg, envs: GymEnv, eval_envs_dict: Dict[str, GymEnv], device: str):
        super().__init__(cfg, envs, eval_envs_dict, device)
        
        # self.l1ac = L1AC(cfg, device)

        self.unit_mass = cfg.l1ac.mass
        self.acc_z_des = None

    def init_aux_vars(self, num_envs):
        super().init_aux_vars(num_envs)
        self.l1ac = L1AC_batch(num_envs, self.cfg.l1ac, self.device)
        self.acc_z_des = None
        
    def __call__(self, pid_obs: Dict[str, Any]) -> Tuple[torch.Tensor]:
        state = pid_obs.get('state') # QuadrotorState
        reference = pid_obs.get('reference') # Dict
        time = pid_obs.get('time') # torch.Tensor
        
        # self.update_dt(time)
        
        pos, vel, quat = state.pos, state.vel, state.quat
        euler_est = ru.quat2euler(quat)
            
        if self.acc_z_des is None:
            self.acc_z_des = torch.zeros_like(vel[..., 2:])
            
        # compute the last propeller thrust in the world frame
        force_t = ru.rotate_axis(2, quat, mode='quat') * self.acc_z_des #* self.unit_mass

        # acc_hat = torch.zeros_like(force_t)
        # if self.use_l1:
        acc_hat = pid_obs.get('wind_vec') #self.l1ac.adaptation_fn(vel, force_t, self.dt) # #
        # if self.count % 10 == 0:
        #     print(f'acc_hat: {acc_hat}')
        self.l1ac_log = [acc_hat[..., 0], acc_hat[..., 1], acc_hat[..., 2]]

        pos_err = pos - reference.get('position', torch.zeros_like(pos))
        vel_err = vel - reference.get('linear_velocity', torch.zeros_like(vel))
        yaw_err = euler_est[..., 2] - torch.zeros_like(euler_est[..., 2])
        roll_err = euler_est[..., 0] - torch.zeros_like(euler_est[..., 0])
        pitch_err = euler_est[..., 1] - torch.zeros_like(euler_est[..., 1])

        def wrap_angle(angle):
            if torch.any(angle >=  torch.pi):
                return torch.where(angle >= torch.pi, angle - 2.0 * torch.pi, angle)
            elif torch.any(angle < -torch.pi):
                return torch.where(angle < -torch.pi, angle + 2.0 * torch.pi, angle)
            else:
                return torch.where((angle >= -torch.pi) & (angle < torch.pi), angle, angle)
            
        yaw_err_wraped = wrap_angle(yaw_err)
        roll_err_wraped = wrap_angle(roll_err)
        pitch_err_wraped = wrap_angle(pitch_err)
        
        if self.pos_err_int is None:
            self.pos_err_int = torch.zeros_like(pos)
        self.pos_err_int = self.pos_err_int + pos_err * self.dt
        self.pos_err_int[..., :2] = torch.clamp(self.pos_err_int[..., :2], -self.int_limit_xy, self.int_limit_xy)
        self.pos_err_int[...,  2] = torch.clamp(self.pos_err_int[...,  2], -self.int_limit_z,  self.int_limit_z )
        
        g_vec = torch.tensor([0, 0, self.g]).float().to(self.device)
        g_vec = g_vec.unsqueeze(0).expand_as(pos)
        
        z_vec = torch.tensor([0, 0, 1]).float().to(self.device)
        z_vec = z_vec.unsqueeze(0).expand_as(pos)

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
        acc_des = acc_des - acc_hat # in world frame
            
        u_des = ru.inv_rotate_vector(acc_des, quat, mode='quat') # world to body
        self.acc_z_des = torch.norm(u_des, dim=-1, keepdim=True)
        
        rot_err = torch.linalg.cross(u_des / self.acc_z_des, z_vec)
        omega_des = -self.kp_rot * rot_err
        euler_feedback_des = torch.zeros_like(omega_des)
        euler_feedback_des[..., 2] = -self.kp_yaw * (yaw_err_wraped - reference.get('heading_rate', torch.zeros_like(omega_des[..., 2])))
        omega_des_yaw = ru.omega_rotate_from_euler(euler_feedback_des, euler_est)

        omega_des[..., 0] = omega_des[..., 0] - self.kp_rot_rp * roll_err_wraped
        omega_des[..., 1] = omega_des[..., 1] - self.kp_rot_rp * pitch_err_wraped
        omega_des[..., 2] = omega_des[..., 2] - self.kp_yaw * yaw_err_wraped

        # Apply limits to action outputs
        # acc_des = torch.clamp(acc_des, 0.0, 2.0 * self.g)  # Limit acceleration between 0 and 1.5G
        # omega_des = torch.clamp(omega_des, -1.57, -1.57)      # Limit angular velocity to ±5 rad/s
        
        self.count += 1
        self.v_prev = vel
        
        return torch.cat([self.acc_z_des, omega_des], dim=-1)
    
register_controller('l1_pid', L1_PID)