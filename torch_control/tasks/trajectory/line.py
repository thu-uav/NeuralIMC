from typing import List, Union

import torch
from torch_control.tasks.trajectory.base import BaseTrajectory


class Line(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 origin: torch.Tensor = torch.zeros(3),
                 line_length: Union[float, List] = 2, 
                 height: Union[float, List] = 0,
                 period: Union[float, List] = 2,
                 device: str = 'cpu',
                 seed: int = 0):
        
        super().__init__(num_trajs, origin, device, seed)
        
        self._line_length = line_length
        self._height = height
        self._period = period
        
        if isinstance(self._line_length, list):
            self.line_length = torch.rand(self.num_trajs, device=self.device) * (
                self._line_length[1] - self._line_length[0]) + self._line_length[0]
        else:
            self.line_length = torch.ones(self.num_trajs, device=self.device) * self._line_length
        if isinstance(self._height, list):
            self.height = torch.rand(self.num_trajs, device=self.device) * (
                self._height[1] - self._height[0]) + self._height[0]
        else:
            self.height = torch.ones(self.num_trajs, device=self.device) * self._height
        if isinstance(self._period, list):
            self.T = torch.rand(self.num_trajs, device=self.device) * (
                self._period[1] - self._period[0]) + self._period[0]
        else:
            self.T = torch.ones(self.num_trajs, device=self.device) * self._period
            
        self.reset()
            
    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)
        
        num_trajs = idx.shape[0]
            
        if isinstance(self._line_length, list):
            self.line_length[idx] = torch.rand(num_trajs, device=self.device) * (
                self._line_length[1] - self._line_length[0]) + self._line_length[0]
        else:
            self.line_length[idx] = torch.ones(num_trajs, device=self.device) * self._line_length
        if isinstance(self._height, list):
            self.height[idx] = torch.rand(num_trajs, device=self.device) * (
                self._height[1] - self._height[0]) + self._height[0]
        else:
            self.height[idx] = torch.ones(num_trajs, device=self.device) * self._height
        if isinstance(self._period, list):
            self.T[idx] = torch.rand(num_trajs, device=self.device) * (
                self._period[1] - self._period[0]) + self._period[0]
        else:
            self.T[idx] = torch.ones(num_trajs, device=self.device) * self._period

    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            T = self.T.unsqueeze(1)
            line_length = self.line_length.unsqueeze(1)
            height = self.height.unsqueeze(1)
        else:
            T = self.T
            line_length = self.line_length
            height = self.height
            
        x = torch.zeros_like(t)
        forward = ((t // T) % 2 == 0).float()  # Convert boolean mask to float for multiplication
        back = ((t // T) % 2 == 1).float()

        # Perform operations with masking
        x += (line_length / T * (t % T)) * forward  # Add forward movement
        x += (line_length - (line_length / T * (t % T))) * back  # Add backward movement

        return torch.stack([x, t*0, t*0 + height], dim=-1).float() + self.origin

    def vel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            T = self.T.unsqueeze(1)
            line_length = self.line_length.unsqueeze(1)
            height = self.height.unsqueeze(1)
        else:
            T = self.T
            line_length = self.line_length
            height = self.height
            
        x = torch.zeros_like(t)
        forward = ((t // T) % 2 == 0).float()
        back = ((t // T) % 2 == 1).float()

        # Perform operations with masking
        x += (line_length / T) * forward  # Add forward velocity
        x -= (line_length / T) * back  # Subtract for backward velocity

        return torch.stack([x, t*0, t*0], dim=-1).float()

if __name__ == '__main__':
    import time

    import matplotlib.pyplot as plt
    import torch_control.utils.rot_utils as ru
    
    line = Line(1, line_length=1.0, period=1.0)
    t = torch.stack([torch.arange(0, 20, 0.1) for _ in range(1)], dim=0)
    pos = line.pos(t).squeeze(0).cpu().numpy()
    vel = line.vel(t).squeeze(0).cpu().numpy()
    quat = line.quat(t)
    ang = ru.quat2euler(quat).squeeze(0).cpu().numpy()
    angvel = line.angvel(t).squeeze(0).cpu().numpy()
    t = t.squeeze(0)
    
    print(pos.shape, vel.shape, quat.shape, angvel.shape)
    # import pdb; pdb.set_trace()
    
    # plot 3D traj and save
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('line.png')
    
    # plot x/y/z and vx/vy/vz and roll/pitch/yaw and omega_x/omega_y/omega_z in 3 * 4 subplots
    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    axs[0,0].plot(t, pos[:,0])
    axs[0,0].set_xlabel('t')
    axs[0,0].set_ylabel('x')
    axs[0,1].plot(t, vel[:,0])
    axs[0,1].set_xlabel('t')
    axs[0,1].set_ylabel('vx')
    axs[0,2].plot(t, ang[:,0])
    axs[0,2].set_xlabel('t')
    axs[0,2].set_ylabel('roll')
    axs[0,3].plot(t, angvel[:,0])
    axs[0,3].set_xlabel('t')
    axs[0,3].set_ylabel('omega_x')
    axs[1,0].plot(t, pos[:,1])
    axs[1,0].set_xlabel('t')
    axs[1,0].set_ylabel('y')
    axs[1,1].plot(t, vel[:,1])
    axs[1,1].set_xlabel('t')
    axs[1,1].set_ylabel('vy')
    axs[1,2].plot(t, ang[:,1])
    axs[1,2].set_xlabel('t')
    axs[1,2].set_ylabel('pitch')
    axs[1,3].plot(t, angvel[:,1])
    axs[1,3].set_xlabel('t')
    axs[1,3].set_ylabel('omega_y')
    axs[2,0].plot(t, pos[:,2])
    axs[2,0].set_xlabel('t')
    axs[2,0].set_ylabel('z')
    axs[2,1].plot(t, vel[:,2])
    axs[2,1].set_xlabel('t')
    axs[2,1].set_ylabel('vz')
    axs[2,2].plot(t, ang[:,2])
    axs[2,2].set_xlabel('t')
    axs[2,2].set_ylabel('yaw')
    axs[2,3].plot(t, angvel[:,2])
    axs[2,3].set_xlabel('t')
    axs[2,3].set_ylabel('omega_z')
    
    plt.tight_layout()
    plt.savefig('line_xyz.png')
    
    
    