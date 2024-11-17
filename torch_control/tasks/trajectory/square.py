from typing import List, Union

import torch
from torch_control.tasks.trajectory.base import BaseTrajectory


class Square(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 origin: torch.Tensor = torch.zeros(3),
                 height: Union[float, List] = 0.0,
                 x_length: Union[float, List] = 2.0,
                 y_length: Union[float, List] = 3.0,
                 x_period: Union[float, List] = 4.0,
                 y_period: Union[float, List] = 5.0,
                 device: str = 'cpu',
                 seed: int = 0):
                
        super().__init__(num_trajs, origin, device, seed)

        self._height = height
        self._x_length = x_length
        self._y_length = y_length
        self._x_period = x_period
        self._y_period = y_period
        
        if isinstance(self._height, list):
            self.height = torch.rand(num_trajs, device=self.device) * (
                self._height[1] - self._height[0]) + self._height[0]
        else:
            self.height = torch.ones(num_trajs, device=self.device) * self._height
        if isinstance(self._x_length, list):
            self.x_length = torch.rand(num_trajs, device=self.device) * (
                self._x_length[1] - self._x_length[0]) + self._x_length[0]
        else:
            self.x_length = torch.ones(num_trajs, device=self.device) * self._x_length
        if isinstance(self._y_length, list):
            self.y_length = torch.rand(num_trajs, device=self.device) * (
                self._y_length[1] - self._y_length[0]) + self._y_length[0]
        else:
            self.y_length = torch.ones(num_trajs, device=self.device) * self._y_length
        if isinstance(self._x_period, list):
            self.x_period = torch.rand(num_trajs, device=self.device) * (
                self._x_period[1] - self._x_period[0]) + self._x_period[0]
        else:
            self.x_period = torch.ones(num_trajs, device=self.device) * self._x_period
        if isinstance(self._y_period, list):
            self.y_period = torch.rand(num_trajs, device=self.device) * (
                self._y_period[1] - self._y_period[0]) + self._y_period[0]
        else:
            self.y_period = torch.ones(num_trajs, device=self.device) * self._y_period
            
        self.reset()
            
    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)

        num_trajs = idx.shape[0]
        
        if isinstance(self._height, list):
            self.height[idx] = torch.rand(num_trajs, device=self.device) * (
                self._height[1] - self._height[0]) + self._height[0]
        else:
            self.height[idx] = torch.ones(num_trajs, device=self.device) * self._height
        if isinstance(self._x_length, list):
            self.x_length[idx] = torch.rand(num_trajs, device=self.device) * (
                self._x_length[1] - self._x_length[0]) + self._x_length[0]
        else:
            self.x_length[idx] = torch.ones(num_trajs, device=self.device) * self._x_length
        if isinstance(self._y_length, list):
            self.y_length[idx] = torch.rand(num_trajs, device=self.device) * (
                self._y_length[1] - self._y_length[0]) + self._y_length[0]
        else:
            self.y_length[idx] = torch.ones(num_trajs, device=self.device) * self._y_length
        if isinstance(self._x_period, list):
            self.x_period[idx] = torch.rand(num_trajs, device=self.device) * (
                self._x_period[1] - self._x_period[0]) + self._x_period[0]
        else:
            self.x_period[idx] = torch.ones(num_trajs, device=self.device) * self._x_period
        if isinstance(self._y_period, list):
            self.y_period[idx] = torch.rand(num_trajs, device=self.device) * (
                self._y_period[1] - self._y_period[0]) + self._y_period[0]
        else:
            self.y_period[idx] = torch.ones(num_trajs, device=self.device) * self._y_period

    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            x_length = self.x_length.unsqueeze(1)
            y_length = self.y_length.unsqueeze(1)
            height = self.height.unsqueeze(1)
            x_period = self.x_period.unsqueeze(1)
            y_period = self.y_period.unsqueeze(1)
        else:
            x_length = self.x_length
            y_length = self.y_length
            height = self.height
            x_period = self.x_period
            y_period = self.y_period
            
        x = torch.zeros_like(t)
        y = torch.zeros_like(t)
        
        c1 = ((t // x_period) % 2 == 0).float()
        c2 = ((t // x_period) % 2 == 1).float()
        x += (x_length / x_period * (t % x_period)) * c1
        x += (x_length - (x_length / x_period * (t % x_period))) * c2

        c1 = ((t // y_period) % 4 == 0).float()
        c2 = ((t // y_period) % 4 == 1).float()
        c3 = ((t // y_period) % 4 == 2).float()
        c4 = ((t // y_period) % 4 == 3).float()
        y += (y_length / y_period * (t % y_period)) * c1
        y += (y_length - (y_length / y_period * (t % y_period))) * c2
        y += (-y_length / y_period * (t % y_period)) * c3
        y += (-y_length + (y_length / y_period * (t % y_period))) * c4
        
        return torch.stack([x, y, t*0 + height], dim=-1).float() + self.origin

    def vel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs, device=self.device) * t
            
        if t.ndim > 1:
            x_length = self.x_length.unsqueeze(1)
            y_length = self.y_length.unsqueeze(1)
            height = self.height.unsqueeze(1)
            x_period = self.x_period.unsqueeze(1)
            y_period = self.y_period.unsqueeze(1)
        else:
            x_length = self.x_length
            y_length = self.y_length
            height = self.height
            x_period = self.x_period
            y_period = self.y_period
            
        x = torch.zeros_like(t)
        y = torch.zeros_like(t)
        
        c1 = ((t // x_period) % 2 == 0).float()
        c2 = ((t // x_period) % 2 == 1).float()
        x += (x_length / x_period) * c1
        x += (-x_length / x_period) * c2
        
        c1 = ((t // y_period) % 4 == 0).float()
        c2 = ((t // y_period) % 4 == 1).float()
        c3 = ((t // y_period) % 4 == 2).float()
        c4 = ((t // y_period) % 4 == 3).float()
        y += (y_length / y_period) * c1
        y += (-y_length / y_period) * c2
        y += (-y_length / y_period) * c3
        y += (y_length / y_period) * c4
            
        return torch.stack([x, y, t*0], dim=-1).float()


# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch_control.utils.rot_utils as ru
    
    ref = Square(1, x_length=1.0, y_length=0.5, x_period=1.0, y_period=0.5)
    t = torch.stack([torch.arange(0, 50, 0.1) for _ in range(1)], dim=0)
    pos = ref.pos(t).squeeze(0).cpu().numpy()
    vel = ref.vel(t).squeeze(0).cpu().numpy()
    acc = ref.acc(t).squeeze(0).cpu().numpy()
    jerk = ref.jerk(t).squeeze(0).cpu().numpy()
    snap = ref.snap(t).squeeze(0).cpu().numpy()
    quat = ref.quat(t)
    ang = ru.quat2euler(quat).squeeze(0).cpu().numpy()
    angvel = ref.angvel(t).squeeze(0).cpu().numpy()
    yaw = ref.yaw(t).squeeze(0)
    yawvel = ref.yawvel(t).squeeze(0)
    yawacc = ref.yawacc(t).squeeze(0)
    yaw_data = torch.stack([yaw, yawvel, yawacc], dim=-1).cpu().numpy()
    t = t.squeeze(0).cpu().numpy()
    
    print(pos.shape, vel.shape, acc.shape, jerk.shape, snap.shape, quat.shape, angvel.shape, yaw.shape, yawvel.shape, yawacc.shape)
    # import pdb; pdb.set_trace()
    
    # plot 3D traj and save
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('square.png')
    
    # plot x/y/z and vx/vy/vz in 3 * 2 subplots
    fig, axs = plt.subplots(3, 8, figsize=(80, 10))
    for i in range(3):
        axs[i,0].plot(t, pos[:,i])
        axs[i,0].set_xlabel('t')
        axs[i,0].set_ylabel('x')
        axs[i,1].plot(t, vel[:,i])
        axs[i,1].set_xlabel('t')
        axs[i,1].set_ylabel('v')
        axs[i,2].plot(t, acc[:,i])
        axs[i,2].set_xlabel('t')
        axs[i,2].set_ylabel('a')
        axs[i,3].plot(t, jerk[:,i])
        axs[i,3].set_xlabel('t')
        axs[i,3].set_ylabel('j')
        axs[i,4].plot(t, snap[:,i])
        axs[i,4].set_xlabel('t')
        axs[i,4].set_ylabel('s')
        axs[i,5].plot(t, ang[:,i])
        axs[i,5].set_xlabel('t')
        axs[i,5].set_ylabel('ang')
        axs[i,6].plot(t, angvel[:,i])
        axs[i,6].set_xlabel('t')
        axs[i,6].set_ylabel('angvel')
        axs[i,7].plot(t, yaw_data[:,i])
        axs[i,7].set_xlabel('t')
        axs[i,7].set_ylabel('yaw')
    
    plt.tight_layout()
    plt.savefig('square_xyz.png')
    
    
    