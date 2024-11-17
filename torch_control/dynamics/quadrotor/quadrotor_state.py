from typing import Any, Dict, List

import torch
import torch.distributions as D
from torch_control.utils import rot_utils as ru


class QuadrotorState(object):
    """
    State of a quadrotor
    - pos: m, in world frame
    - vel: m/s, in world frame
    - acc: m/s^2, in world frame
    - quat: (w, x, y, z), in world frame
    - ang_vel: rad/s, in body frame
    - ang_acc: rad/s^2, in body frame
    - body_torque: Nm, in body frame
    - motor_angvel: rad/s, in motor frame 
    """
    full_state: List = ['pos', 'vel', 'acc', 'quat', 'ang_vel', 'body_torque', 'motor_angvel']

    def __init__(self, num_instances: int = 1, **kwargs):
        if len(kwargs) == 0:
            # Initialize with num_instances
            self.N = num_instances
            self.pos = torch.zeros(self.N, 3)
            self.vel = torch.zeros(self.N, 3)
            self.acc = torch.zeros(self.N, 3)
            self.quat = torch.tensor([[1., 0., 0., 0.]]).repeat(self.N, 1)
            self.ang_vel = torch.zeros(self.N, 3)
            self.body_torque = torch.zeros(self.N, 3)
            self.motor_angvel = torch.zeros(self.N, 4)
        else:
            assert len(kwargs) <= 7, f"QuadrotorState takes 7 arguments, got {len(kwargs)}"
            # Initialize with kwargs
            self.N = len(kwargs['pos'])
            self.pos = kwargs['pos']
            self.vel = kwargs.get('vel', torch.zeros(self.N, 3))
            self.acc = kwargs.get('acc', torch.zeros(self.N, 3))
            self.quat = kwargs.get('quat', torch.zeros(self.N, 4))
            self.ang_vel = kwargs.get('ang_vel', torch.zeros(self.N, 3))
            self.body_torque = kwargs.get('body_torque', torch.zeros(self.N, 3))
            self.motor_angvel = kwargs.get('motor_angvel', torch.zeros(self.N, 4))

    @classmethod
    def construct_from_vec(self, state_vec: torch.Tensor):
        pos, vel, quat, ang_vel = state_vec.split([3, 3, 4, 3], dim=-1)
        return QuadrotorState(
            pos=pos,
            vel=vel,
            quat=quat,
            ang_vel=ang_vel
        )

    def to_vec(self, bodyrate: bool = False):
        if bodyrate:
            return torch.cat([self.pos, self.vel, self.quat, self.ang_vel,], dim=-1)
        else:
            return torch.cat([self.pos, self.vel, self.quat], dim=-1)

    def to(self, device):
        self.device = device
        self.pos = self.pos.to(device)
        self.vel = self.vel.to(device)
        self.acc = self.acc.to(device)
        self.quat = self.quat.to(device)
        self.ang_vel = self.ang_vel.to(device)
        self.body_torque = self.body_torque.to(device)
        self.motor_angvel = self.motor_angvel.to(device)
        return self
    
    def reset(self, 
              idx: torch.Tensor = None, 
              randomization: Dict[str, D.Distribution] = None,
              state_vec: torch.Tensor = None):
        if idx is None:
            idx = torch.arange(self.N)

        if randomization is None or state_vec is not None:
            randomization = {}

        for item in self.full_state:
            if item in randomization.keys():
                perturb_dist = randomization[item]
                perturbed_value = perturb_dist.sample((len(idx),)).float().to(self.device)
                if item == 'ang':
                    perturbed_value = ru.euler2quat(perturbed_value * torch.pi / 180.)
                    getattr(self, 'quat')[idx] = perturbed_value
                elif item == 'ang_vel':
                    getattr(self, item)[idx] = perturbed_value * torch.pi / 180.
                else:
                    getattr(self, item)[idx] = perturbed_value
            else:
                getattr(self, item)[idx] = torch.zeros_like(getattr(self, item)[idx])
                if item == 'quat':
                    getattr(self, item)[idx, 0] = 1.

        if state_vec is not None:
            pos, vel, quat, ang_vel = state_vec.split([3, 3, 4, 3], dim=-1)
            self.pos[idx] = pos
            self.quat[idx] = quat
            self.vel[idx] = vel
            self.ang_vel[idx] = ang_vel
            print("===== Forcibly reset state =====")
            print(self)
            print("=============================")

        self.quat = self.quat / self.quat.norm(dim=-1, keepdim=True)

    def set(self, attr, value):
        assert hasattr(self, attr), f"QuadrotorState has no attribute {attr}"
        setattr(self, attr, value)
        
    def get(self, attr):
        assert hasattr(self, attr), f"QuadrotorState has no attribute {attr}"
        return getattr(self, attr)

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return QuadrotorState(
            pos=self.pos[idx],
            vel=self.vel[idx],
            acc=self.acc[idx],
            quat=self.quat[idx],
            ang_vel=self.ang_vel[idx],
            body_torque=self.body_torque[idx],
            motor_angvel=self.motor_angvel[idx]
        ).to(self.device)
        
    def __setitem__(self, idx, value):
        self.pos[idx] = value.pos
        self.vel[idx] = value.vel
        self.acc[idx] = value.acc
        self.quat[idx] = value.quat
        self.ang_vel[idx] = value.ang_vel
        self.body_torque[idx] = value.body_torque
        self.motor_angvel[idx] = value.motor_angvel

    def __add__(self, d_state):
        return QuadrotorState(
            pos=self.pos + d_state.pos,
            vel=self.vel + d_state.vel,
            acc=self.acc + d_state.acc,
            quat=self.quat + d_state.quat,
            ang_vel=self.ang_vel + d_state.ang_vel,
            body_torque=self.body_torque + d_state.body_torque,
            motor_angvel=self.motor_angvel + d_state.motor_angvel
        ).to(self.device)
    
    def __mul__(self, scalar: float):
        return QuadrotorState(
            pos=self.pos * scalar,
            vel=self.vel * scalar,
            acc=self.acc * scalar,
            quat=self.quat * scalar,
            ang_vel=self.ang_vel * scalar,
            body_torque=self.body_torque * scalar,
            motor_angvel=self.motor_angvel * scalar
        ).to(self.device)
    
    def __rmul__(self, scalar: float):
        return QuadrotorState(
            pos=self.pos * scalar,
            vel=self.vel * scalar,
            acc=self.acc * scalar,
            quat=self.quat * scalar,
            ang_vel=self.ang_vel * scalar,
            body_torque=self.body_torque * scalar,
            motor_angvel=self.motor_angvel * scalar
        ).to(self.device)
    
    def clone(self):
        return QuadrotorState(
            pos=self.pos.clone(),
            vel=self.vel.clone(),
            acc=self.acc.clone(),
            quat=self.quat.clone(),
            ang_vel=self.ang_vel.clone(),
            body_torque=self.body_torque.clone(),
            motor_angvel=self.motor_angvel.clone()
        ).to(self.device)
    
    def __repr__(self) -> str:
        return f"""QuadrotorState(
            pos={self.pos} (m),
            vel={self.vel} (m/s),
            acc={self.acc} (m/s^2),
            quat={self.quat} (w, x, y, z),
            ang_vel={self.ang_vel} (rad/s),
            body_torque={self.body_torque} (Nm),
            motor_angvel={self.motor_angvel} (rad/s)
        )"""