from typing import Union

import numpy as np
import torch
from torch_control.utils import rot_utils as ru
from torch_control.utils.common import set_all_seed

class ReferenceState():
    def __init__(self, 
                 pos=torch.zeros(3), 
                 vel=torch.zeros(3),
                 acc = torch.zeros(3),
                 jerk = torch.zeros(3), 
                 snap = torch.zeros(3),
                 rot=torch.tensor([1.,0.,0.,0.]),
                 ang=torch.zeros(3)):
        
        self.pos = pos # R^3
        self.vel = vel # R^3
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
        self.ang = ang # R^3
        self.t = 0.

class BaseTrajectory():
    def __init__(self, 
                 num_trajs: int = 1, 
                 origin: torch.Tensor = None,
                 device: str = 'cpu',
                 seed: int = 0):
        self.num_trajs = num_trajs
        if origin is not None:
            self.origin = torch.tensor(list(origin)).float().to(device)
        else:
            self.origin = torch.zeros(self.num_trajs, 3).float().to(device)
        self.device = device
        self.seed = seed
        set_all_seed(seed)
    
    def reset(self,
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        if idx is None:
            idx = torch.arange(self.num_trajs, device=self.device)

        if origin is not None:
            assert origin.shape == (len(idx), 3), f"Origin must be a tensor of shape ({len(idx)}, 3), but got {origin.shape}"
            self.origin[idx] = origin
            if verbose:
                print("===== Forcibly reset reference trajectory =====")
                print(f"[{self.__class__}] origin (meter): {self.origin[idx].tolist()} for idx: {idx.tolist()}")
                print("=============================")

        return idx

    def get_ref_tensor(self, t: Union[float, torch.Tensor]):
        pos = self.pos(t)
        quat = self.quat(t)
        vel = self.vel(t)
        omega = self.angvel(t)

        ref_tensor = torch.stack([pos, quat, vel, omega], dim=-1)
        return ref_tensor

    def get_state_struct(self, t: Union[float, torch.Tensor]):
        return ReferenceState(
            pos = self.pos(t),
            vel = self.vel(t),
            acc = self.acc(t),
            jerk = self.jerk(t),
            snap = self.snap(t),
            rot = self.quat(t),
            ang = self.angvel(t),
        )
        
    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            p = torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            p = torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
            
        return p + self.origin
    
    def vel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
    
    def acc(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)

    def jerk(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)

    def snap(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
        
    def quat(self, t: Union[float, torch.Tensor]):
        '''
        w,x,y,z
        '''
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t**0, t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t**0, t*0, t*0, t*0], dim=-1).float().to(self.device)
        
    def angvel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
        
    def yaw(self, t: Union[float, torch.Tensor]):
        return t * 0.

    def yawvel(self, t: Union[float, torch.Tensor]):
        return t * 0.

    def yawacc(self, t: Union[float, torch.Tensor]):
        return t * 0.
    
    def euler_ang(self, t: Union[float, torch.Tensor]):
        yaw = self.yaw(t)
        zero_ang = torch.zeros_like(yaw)
        return torch.stack([zero_ang, zero_ang, yaw], dim=-1)
    
    def plot(self):
        import matplotlib.pyplot as plt
        import time

        datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    
        t = torch.stack([torch.arange(0, 20, 0.1) 
                         for _ in range(self.num_trajs)], dim=0).to(self.device)
        
        lists = {k: [] for k in ['pos', 'vel', 'acc', 'jerk', 'snap', 'ang', 'angvel', 'yaw_data']}

        for t_idx in range(t.shape[1]):
            tt = t[:, t_idx]
            pos = self.pos(tt).cpu().numpy()
            vel = self.vel(tt).cpu().numpy()
            acc = self.acc(tt).cpu().numpy()
            jerk = self.jerk(tt).cpu().numpy()
            snap = self.snap(tt).cpu().numpy()
            quat = self.quat(t)
            ang = ru.quat2euler(quat)
            ang = ang.cpu().numpy()
            angvel = self.angvel(tt).cpu().numpy()
            yaw = self.yaw(tt)
            yawvel = self.yawvel(tt)
            yawacc = self.yawacc(tt)
            yaw_data = torch.stack([yaw, yawvel, yawacc], dim=-1).cpu().numpy()

            lists['pos'].append(pos)
            lists['vel'].append(vel)
            lists['acc'].append(acc)
            lists['jerk'].append(jerk)
            lists['snap'].append(snap)
            lists['ang'].append(ang)
            lists['angvel'].append(angvel)
            lists['yaw_data'].append(yaw_data)

        pos = np.stack(lists['pos'], axis=1)
        vel = np.stack(lists['vel'], axis=1)
        acc = np.stack(lists['acc'], axis=1)
        jerk = np.stack(lists['jerk'], axis=1)
        snap = np.stack(lists['snap'], axis=1)
        ang = np.stack(lists['ang'], axis=1)
        angvel = np.stack(lists['angvel'], axis=1)
        yaw_data = np.stack(lists['yaw_data'], axis=1)

        for idx in range(min(10, self.num_trajs)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(pos[idx, :,0], pos[idx, :,1], pos[idx, :,2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.savefig(f'figs/{self.__class__.__name__}-seed{self.seed}-{idx}-{datetime}.png')

            fig, axs = plt.subplots(3, 8, figsize=(80, 10))
            for i in range(3):
                axs[i,0].plot(np.arange(0, 20, 0.1), pos[idx,:,i])
                axs[i,0].set_xlabel('t')
                axs[i,0].set_ylabel('x')
                axs[i,1].plot(np.arange(0, 20, 0.1), vel[idx,:,i])
                axs[i,1].set_xlabel('t')
                axs[i,1].set_ylabel('v')
                axs[i,2].plot(np.arange(0, 20, 0.1), acc[idx,:,i])
                axs[i,2].set_xlabel('t')
                axs[i,2].set_ylabel('a')
                axs[i,3].plot(np.arange(0, 20, 0.1), jerk[idx,:,i])
                axs[i,3].set_xlabel('t')
                axs[i,3].set_ylabel('j')
                axs[i,4].plot(np.arange(0, 20, 0.1), snap[idx,:,i])
                axs[i,4].set_xlabel('t')
                axs[i,4].set_ylabel('s')
                axs[i,5].plot(np.arange(0, 20, 0.1), ang[idx,:,i])
                axs[i,5].set_xlabel('t')
                axs[i,5].set_ylabel('ang')
                axs[i,6].plot(np.arange(0, 20, 0.1), angvel[idx,:,i])
                axs[i,6].set_xlabel('t')
                axs[i,6].set_ylabel('angvel')
                axs[i,7].plot(np.arange(0, 20, 0.1), yaw_data[idx,:,i])
                axs[i,7].set_xlabel('t')
                axs[i,7].set_ylabel('yaw')

            plt.tight_layout()
            plt.savefig(f'figs/{self.__class__.__name__}_xyz-seed{self.seed}-{idx}-{datetime}.png')

if __name__=='__main__':
    a = BaseTrajectory()
    t = torch.tensor([0., 1.0])
    a.ref_vec(t)
