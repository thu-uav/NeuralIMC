from typing import List, Union

import torch
try:
    from .base import BaseTrajectory
except:
    from base import BaseTrajectory
from torch_control.utils import math_utils as mu


class ChainedPolynomial(BaseTrajectory):
    def __init__(self,
                 num_trajs: int,
                 scale: float = 1.5,
                 use_y: bool = True,
                 min_dt: float = 1.5,
                 max_dt: float = 4.0,
                 degree: int = 5,
                 origin: torch.Tensor = torch.zeros(3),
                 device: str = 'cpu',
                 seed: int = 0):
        super().__init__(num_trajs, origin, device, seed)
        assert degree % 2 == 1

        self.use_y = use_y
        self.degree = degree
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.scale = scale

        self.num_segments = 200

        self.x_coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, 
                                    dtype=torch.float32, device=self.device)
        self.T_x = torch.zeros(self.num_trajs, self.num_segments + 1, 
                               dtype=torch.float32, device=self.device)
        if self.use_y:
            self.y_coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, 
                                        dtype=torch.float32, device=self.device)
            self.T_y = torch.zeros(self.num_trajs, self.num_segments + 1, 
                                   dtype=torch.float32, device=self.device)
        self.reset()

    def generate_coeffs(self, dt: torch.Tensor):
        b_values = torch.rand(self.num_trajs, (self.degree + 1) // 2, self.num_segments + 1, 
                              dtype=torch.float32, device=self.device) * self.scale * 2 - self.scale # (num_trajs, (degree + 1) // 2, num_segments + 1)
        b_values[:, 0] = 0.
        b_values = torch.cat([b_values[..., :-1], b_values[..., 1:]], dim=1) # (num_trajs, degree + 1, num_segments)

        A_values = torch.zeros(self.num_trajs, self.degree + 1, self.degree + 1, self.num_segments, dtype=torch.float32, device=self.device)
        coeffs = torch.zeros(self.num_trajs, self.degree + 1, self.num_segments, dtype=torch.float32, device=self.device)

        for i in range(self.num_segments):
            A_values[..., i] = self.deriv_fitting_matrix(dt[:, i])
            coeffs[..., i] = torch.linalg.solve(A_values[..., i], b_values[..., i])

        coeffs = coeffs.flip(dims=[-2])

        return coeffs

    def deriv_fitting_matrix(self, dt: torch.Tensor, degree: int = None):
        if degree is None:
            degree = self.degree + 1

        A = torch.zeros(self.num_trajs, degree, degree, device=self.device)

        ts = dt[:, None] ** torch.arange(degree, device=self.device) # (num_trajs, degree)
        constant_term = 1.
        poly = torch.ones(self.num_trajs, degree, device=self.device) # (num_trajs, degree)

        for i in range(degree // 2):
            A[:, i, i] = constant_term
            A[:, i + degree // 2, :] = torch.cat([
                torch.zeros(self.num_trajs, i, device=self.device),
                ts[:, :degree - i] * poly[:, :degree - i],
            ], dim=1)
            poly = mu.polyder(poly, increasing_order=True)
            constant_term *= i + 1

        return A

    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)

        dt_x = torch.rand(self.num_trajs, self.num_segments, device=self.device
                          ) * (self.max_dt - self.min_dt) + self.min_dt # (num_trajs, num_segments)
        T_x = torch.cat([torch.cumsum(dt_x, dim=1),
                         torch.zeros(self.num_trajs, 1, device=self.device),
                         ], dim=1).contiguous()
        x_coeffs = self.generate_coeffs(dt_x) # (num_trajs, degree + 1, num_segments)
        self.x_coeffs[idx] = x_coeffs[idx]
        self.T_x[idx] = T_x[idx]

        if self.use_y:
            dt_y = torch.rand(self.num_trajs, self.num_segments, device=self.device
                              ) * (self.max_dt - self.min_dt) + self.min_dt
            T_y = torch.cat([torch.cumsum(dt_y, dim=1),
                             torch.zeros(self.num_trajs, 1, device=self.device),
                             ], dim=1).contiguous()
            y_coeffs = self.generate_coeffs(dt_y)
            self.y_coeffs[idx] = y_coeffs[idx]
            self.T_y[idx] = T_y[idx]

    def pos(self, t: torch.Tensor):
        # assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1) # (num_trajs,)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1] # (num_trajs,)
        x = mu.poly(self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 
                    (t - offset)[:, None],)

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 
                        (t - offset)[:, None],)
        else:
            y = x * 0.

        z = x * 0.


        return torch.cat([x, y, z], dim=-1) + self.origin

    def vel(self, t: torch.Tensor):
        # assert t.shape == (self.num_trajs,)
        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x]), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y]), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    
    def acc(self, t: torch.Tensor):
        # assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 2), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 2), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)

    def jerk(self, t: torch.Tensor):
        # assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 3), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 3), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    
    def snap(self, t: torch.Tensor):
        # assert t.shape == (self.num_trajs,)

        idx_x = torch.searchsorted(self.T_x, t[:, None]).squeeze(-1)
        offset = self.T_x[torch.arange(self.num_trajs, device=self.device), idx_x - 1]
        x = mu.poly(mu.polyder(
            self.x_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_x], 4), 
            (t - offset)[:, None])

        if self.use_y:
            idx_y = torch.searchsorted(self.T_y, t[:, None]).squeeze(-1)
            offset = self.T_y[torch.arange(self.num_trajs, device=self.device), idx_y - 1]
            y = mu.poly(mu.polyder(
                self.y_coeffs[torch.arange(self.num_trajs, device=self.device), :, idx_y], 4), 
                (t - offset)[:, None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    

if __name__ == "__main__":
    ref = ChainedPolynomial(1, min_dt=1.5, max_dt=4.0, degree=5, use_y=True)

    t = torch.arange(0, 40, 0.1, dtype=torch.float32)

    pos = []
    vel = []
    acc = []
    jerk = []
    snap = []
    for tt in t:
        pos.append(ref.pos(tt[None]))
        vel.append(ref.vel(tt[None]))
        acc.append(ref.acc(tt[None]))
        jerk.append(ref.jerk(tt[None]))
        snap.append(ref.snap(tt[None]))

    pos = torch.stack(pos, dim=1).cpu().numpy()
    vel = torch.stack(vel, dim=1).cpu().numpy()
    acc = torch.stack(acc, dim=1).cpu().numpy()
    jerk = torch.stack(jerk, dim=1).cpu().numpy()
    snap = torch.stack(snap, dim=1).cpu().numpy()

    plot_idx = 0
    import matplotlib.pyplot as plt
    import time
    
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[plot_idx, :,0], pos[plot_idx, :,1], pos[plot_idx, :,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(f'figs/chainedpoly-{datetime}.png')

    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    for i in range(3):
        axs[i,0].plot(t, pos[plot_idx, :,i])
        axs[i,0].set_xlabel('t')
        axs[i,0].set_ylabel('x')
        axs[i,1].plot(t, vel[plot_idx, :,i])
        axs[i,1].set_xlabel('t')
        axs[i,1].set_ylabel('v')
    plt.tight_layout()
    plt.savefig(f'figs/chainedpoly_xyz-{datetime}.png')
