import math
from typing import Dict, List, Tuple

import torch
import torch.distributions as D
from torch_control.utils import integrators
from torch_control.utils import math_utils as mu
from torch_control.utils import rot_utils as ru


class RigidBody(object):
    
    def __init__(self, dt: float=0.02):
        self.g = 9.81
        self.dt = dt
    
    def step(self, state_vec, action_vec):
        if state_vec.shape[-1] == 10:
            pos, vel, quat = state_vec.split([3, 3, 4], dim=-1)
        else:
            pos, vel, quat, angvel = state_vec.split([3, 3, 4, 3], dim=-1)
            
        normalized_thrust, bodyrate = action_vec.split([1, 3], dim=-1)
        
        g_vec = torch.tensor([0., 0., -self.g]).to(pos.device)
        
        prop_acc = normalized_thrust * torch.tensor([0., 0., 1.]).to(normalized_thrust.device)
        
        accel = ru.rotate_vector(prop_acc, quat, 'quat') + g_vec # world frame
        
        new_pos = pos + integrators.euler_integral(self.dt, vel, accel)
        new_vel = vel + accel * self.dt
        
        dang = integrators.euler_integral(self.dt, bodyrate, 0.)
        new_quat = integrators.so3_quat_integral(quat, dang, dang_in_body=True)
        new_quat = new_quat / torch.norm(new_quat, dim=-1, keepdim=True)
        
        return torch.cat([new_pos, new_vel, new_quat], dim=-1)
    
    def __call__(self, x, u):
        if x.shape[-1] == 10:
            return self.step(x, u)
        elif x.shape[-1] == 13:
            # directly pass through the bodyrate
            pos_vel_rot = x[..., :10]
            new_pos_vel_rot = self.step(pos_vel_rot, u)
            return torch.cat([new_pos_vel_rot, u[..., 1:]], dim=-1)
        else:
            raise ValueError(f'Unknown state shape {x.shape}')