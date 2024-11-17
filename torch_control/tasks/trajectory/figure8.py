from typing import List, Union

import torch
from torch_control.tasks.trajectory.base import BaseTrajectory


class Figure8(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 origin: torch.Tensor = torch.zeros(3),
                 xy_scale: Union[float, List] = 1.0,
                 z_scale: Union[float, List] = 0.0,
                 period: Union[float, List] = 2*torch.pi,
                 device: str = 'cpu',
                 seed: int = 0):
        super().__init__(num_trajs, origin, device, seed)
        
        self._xy_scale = xy_scale
        self._z_scale = z_scale
        self._period = period
        
        if isinstance(self._xy_scale, list):
            self.xy_scale = torch.rand(self.num_trajs, device=self.device) * (
                self._xy_scale[1] - self._xy_scale[0]) + self._xy_scale[0]
        else:
            self.xy_scale = torch.ones(self.num_trajs, device=self.device) * self._xy_scale
        if isinstance(self._z_scale, list):
            self.z_scale = torch.rand(self.num_trajs, device=self.device) * (
                self._z_scale[1] - self._z_scale[0]) + self._z_scale[0]
        else:
            self.z_scale = torch.ones(self.num_trajs, device=self.device) * self._z_scale
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
        
        if isinstance(self._xy_scale, list):
            self.xy_scale[idx] = torch.rand(num_trajs, device=self.device) * (
                self._xy_scale[1] - self._xy_scale[0]) + self._xy_scale[0]
        else:
            self.xy_scale[idx] = torch.ones(num_trajs, device=self.device) * self._xy_scale
        if isinstance(self._z_scale, list):
            self.z_scale[idx] = torch.rand(num_trajs, device=self.device) * (
                self._z_scale[1] - self._z_scale[0]) + self._z_scale[0]
        else:
            self.z_scale[idx] = torch.ones(num_trajs, device=self.device) * self._z_scale
        if isinstance(self._period, list):
            self.c[idx] = torch.rand(num_trajs, device=self.device) * (
                self._period[1] - self._period[0]) + self._period[0]
            self.c[idx] /= (2 * torch.pi)
        else:
            self.c[idx] = torch.ones(num_trajs, device=self.device) * self._period / (2 * torch.pi)

    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, float):
            t = torch.tensor(t, device=self.device)
        t = t.unsqueeze(-1)
        
        x = self.xy_scale * torch.cos(self.c * t) / (1 + torch.sin(self.c * t)**2)
        y = self.xy_scale * torch.cos(self.c * t) * torch.sin(self.c * t) / (1 + torch.sin(self.c * t)**2)
        z = self.z_scale * torch.sin(self.c * t)
        
        return x, y, z

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch_control.utils.rot_utils as ru
    
    t = torch.linspace(0, 2*torch.pi, 100)
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    # 参数设置
    a = 1  # 控制轨迹大小
    b = 1  # 控制在z轴上的运动幅度
    k = 1  # 控制飞行速度及z轴运动频率

    # 时间范围
    t = np.linspace(0, 2*np.pi, 1000)

    # 参数方程
    x = a * np.cos(k * t) / (1 + np.sin(k * t)**2)
    y = a * np.cos(k * t) * np.sin(k * t) / (1 + np.sin(k * t)**2)
    z = b * np.sin(k * t)
    
    vx = -a * np.sin(k * t) / (1 + np.sin(k * t)**2) - 2 * a * np.cos(k * t)**2 * np.sin(k * t) / (1 + np.sin(k * t)**2)**2
    vy = a * np.cos(k * t)**2 / (1 + np.sin(k * t)**2) - a * np.sin(k * t)**2 / (1 + np.sin(k * t)**2) + 2 * a * np.cos(k * t)**2 * np.sin(k * t)**2 / (1 + np.sin(k * t)**2)**2
    vz = b * k * np.cos(k * t)
    
    def lemniscate(t, c):
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        sin2p1 = torch.square(sin_t) + 1

        x = torch.stack([
            cos_t, sin_t * cos_t, c * sin_t
        ], dim=-1) / sin2p1.unsqueeze(-1)

        return x

    # 绘制轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal', 'box')
    ax.set_title('Lemniscatus Flight Trajectory')
    plt.savefig('Lemniscatus.png', dpi=300)
    
    # plot traj in 2D and save
    fig = plt.figure()
    ax = fig.add_subplot(331)
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(332)
    ax.plot(x, z)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(333)
    ax.plot(y, z)
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(334)
    ax.plot(t, x)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(335)
    ax.plot(t, y)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(336)
    ax.plot(t, z)
    ax.set_xlabel('t')
    ax.set_ylabel('z')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(337)
    ax.plot(t, vx)
    ax.set_xlabel('t')
    ax.set_ylabel('vx')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(338)
    ax.plot(t, vy)
    ax.set_xlabel('t')
    ax.set_ylabel('vy')
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(339)
    ax.plot(t, vz)
    ax.set_xlabel('t')
    ax.set_ylabel('vz')
    ax.set_aspect('equal', 'box')
    fig.suptitle('Lemniscatus Flight Trajectory')
    plt.tight_layout()
    plt.savefig('Lemniscatus_2D.png', dpi=300)
    
    # plot velocity 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, np.sqrt(vx**2 + vy**2 + vz**2))
    ax.set_xlabel('t')
    ax.set_ylabel('velocity')
    ax.set_aspect('equal', 'box')
    fig.suptitle('Lemniscatus Flight Trajectory')
    plt.tight_layout()
    plt.savefig('Lemniscatus_velocity.png', dpi=300)