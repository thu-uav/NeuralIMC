from typing import List, Union

import torch
try:
    from .base import BaseTrajectory
except:
    from base import BaseTrajectory


class Circle(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 origin: torch.Tensor = torch.zeros(3),
                 radius: Union[float, List] = 1.0,
                 height: Union[float, List] = 0.0,
                 period: Union[float, List] = 2*torch.pi,
                 device: str = 'cpu',
                 seed: int = 0):
        super().__init__(num_trajs, origin, device, seed)
        
        self._radius = radius
        self._height = height
        self._period = period
        self.time_to_start = 1.0
        
        if isinstance(self._radius, list):
            self.radius = torch.rand(self.num_trajs, device=self.device) * (
                self._radius[1] - self._radius[0]) + self._radius[0]
        else:
            self.radius = torch.ones(self.num_trajs, device=self.device) * self._radius
        if isinstance(self._height, list):
            self.height = torch.rand(self.num_trajs, device=self.device) * (
                self._height[1] - self._height[0]) + self._height[0]
        else:
            self.height = torch.ones(self.num_trajs, device=self.device) * self._height
        if isinstance(self._period, list):
            self.c = torch.rand(self.num_trajs, device=self.device) * (
                self._period[1] - self._period[0]) + self._period[0]
            self.c /= (2 * torch.pi)
        else:
            self.c = torch.ones(self.num_trajs, device=self.device) * self._period / (2 * torch.pi)
            
        self.reset()
            
    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)
            
        num_trajs = idx.shape[0]
        
        if isinstance(self._radius, list):
            self.radius[idx] = torch.rand(num_trajs, device=self.device) * (
                self._radius[1] - self._radius[0]) + self._radius[0]
        else:
            self.radius[idx] = torch.ones(num_trajs, device=self.device) * self._radius
        if isinstance(self._height, list):
            self.height[idx] = torch.rand(num_trajs, device=self.device) * (
                self._height[1] - self._height[0]) + self._height[0]
        else:
            self.height[idx] = torch.ones(num_trajs, device=self.device) * self._height
        if isinstance(self._period, list):
            self.c[idx] = torch.rand(num_trajs, device=self.device) * (
                self._period[1] - self._period[0]) + self._period[0]
            self.c[idx] /= (2 * torch.pi)
        else:
            self.c[idx] = torch.ones(num_trajs, device=self.device) * self._period / (2 * torch.pi)

    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
            
        if t.ndim > 1:
            radius = self.radius.unsqueeze(1)
            c = self.c.unsqueeze(1)
            height = self.height.unsqueeze(1)
            origin = self.origin.unsqueeze(1)
        else:
            radius = self.radius
            c = self.c
            height = self.height
            origin = self.origin
        return torch.stack([-radius * torch.cos(t / c) + radius,
                            radius * torch.sin(t / c),
                            t*0 + height], dim=-1) + origin

    def vel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            radius = self.radius.unsqueeze(1)
            c = self.c.unsqueeze(1)
            height = self.height.unsqueeze(1)
        else:
            radius = self.radius
            c = self.c
            height = self.height
            
        return torch.stack([radius / c * torch.sin(t / c),
                            radius / c * torch.cos(t / c),
                            t*0], dim=-1)

    def acc(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            radius = self.radius.unsqueeze(1)
            c = self.c.unsqueeze(1)
            height = self.height.unsqueeze(1)
        else:
            radius = self.radius
            c = self.c
            height = self.height
            
        return torch.stack([radius / (c**2) * torch.cos(t / c),
                            -radius / (c**2) * torch.sin(t / c),
                            t*0], dim=-1)

    def jerk(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            radius = self.radius.unsqueeze(1)
            c = self.c.unsqueeze(1)
            height = self.height.unsqueeze(1)
        else:
            radius = self.radius
            c = self.c
            height = self.height
            
        return torch.stack([-radius / (c**3) * torch.sin(t / c),
                            -radius / (c**3) * torch.cos(t / c),
                            t*0], dim=-1)

    def snap(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            radius = self.radius.unsqueeze(1)
            c = self.c.unsqueeze(1)
            height = self.height.unsqueeze(1)
        else:
            radius = self.radius
            c = self.c
            height = self.height
            
        return torch.stack([-radius / (c**4) * torch.cos(t / c),
                            radius / (c**4) * torch.sin(t / c),
                            t*0], dim=-1)

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch_control.utils.rot_utils as ru
    import time
    
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    
    ref = Circle(num_trajs=1, radius=0.5, height=0.0, period=2)
    t = torch.arange(0, 40, 0.1, dtype=torch.float32)

    pos = []
    vel = []
    for tt in t:
        pos.append(ref.pos(tt[None]))
        vel.append(ref.vel(tt[None]))
    pos = torch.cat(pos, dim=0).cpu().numpy()
    vel = torch.cat(vel, dim=0).cpu().numpy()
    t = t.squeeze(0).cpu().numpy()

    # plot 3D traj and save
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(f'figs/circle-{datetime}.png')
    
    # plot x/y/z and vx/vy/vz in 3 * 2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    for i in range(3):
        axs[i,0].plot(t, pos[:,i])
        axs[i,0].set_xlabel('t')
        axs[i,0].set_ylabel('x')
        axs[i,1].plot(t, vel[:,i])
        axs[i,1].set_xlabel('t')
        axs[i,1].set_ylabel('v')
    
    plt.tight_layout()
    plt.savefig(f'figs/circle_xyz-{datetime}.png')
    
    
    