from typing import Callable

import torch
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.utils.rot_utils import quat_multiply


def euler_step(dynamics_fn: Callable, state: QuadrotorState, dt: float):
    deriative = dynamics_fn(state)

    next_state = state + deriative * dt

    return next_state

def rk4_step(dynamics_fn: Callable, state: QuadrotorState, dt: float):
    k1 = dynamics_fn(state)
    k2 = dynamics_fn(state + k1 * dt * 0.5)
    k3 = dynamics_fn(state + k2 * dt * 0.5)
    k4 = dynamics_fn(state + k3 * dt)

    w = torch.tensor([1, 2, 2, 1]) * dt / 6.

    next_state = state + (k1 * w[0] + k2 * w[1] + k3 * w[2] + k4 * w[3])
    return next_state

def euler_integral(dt, vel, accel):
    return vel * dt + 0.5 * accel * dt ** 2

def exp_euler2quat(ang: torch.Tensor) -> torch.Tensor:
    ang_norm = torch.linalg.norm(ang, dim=-1, keepdim=True)
    half_ang = ang_norm / 2
    mask = (ang_norm < 1e-8).float()
    scale = (0.5 + ang_norm ** 2 / 48) * mask + torch.sin(half_ang) / (ang_norm + 1e-8) * (1 - mask)

    return torch.cat([torch.cos(half_ang), scale * ang], dim=-1)

def so3_quat_integral(quat: torch.Tensor, dang: torch.Tensor, dang_in_body: bool) -> torch.Tensor:
    rot_quat = exp_euler2quat(dang)
    if dang_in_body:
        return quat_multiply(quat, rot_quat)
    else:
        return quat_multiply(rot_quat, quat)
