from typing import Union
import torch, sys, os
import pandas as pd
import numpy as np
from torch_control.tasks.trajectory.base import BaseTrajectory


class TrajFile(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 origin: torch.Tensor = torch.zeros(3),
                 path: str = './traj_files/traj_dict_period_11_s.npy', 
                 time_period: int = sys.maxsize, 
                 device: str = 'cpu',
                 seed: int = 0):
        super().__init__(num_trajs, origin, device, seed)
        self._path = path
        self._time_period = time_period
        self._df = None

        self.reset()
            
    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        super().reset(idx, origin, verbose)
        
        try:
            self._df = pd.DataFrame(np.load(self._path, allow_pickle=True).item()['points'])
        except:
            print('[error]')

        if self._df is None or self._df.empty:
            current_path = os.getcwd()
            print(current_path)
            print(f'[error] Trajectory file error! Check here: {self._path}')
            sys.exit()

    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            pos_ref = self._df[time_diff == time_diff.min()]
            if pos_ref.shape[0] > 1:
                pos_ref = pos_ref.iloc[0, :]
        else:
            pos_ref = self._df.tail(1)

        px_ref = (torch.ones(self.num_trajs, device=self.device) * float(pos_ref.iloc[0]['pose_position_x']))
        py_ref = (torch.ones(self.num_trajs, device=self.device) * float(pos_ref.iloc[0]['pose_position_y']))
        pz_ref = (torch.ones(self.num_trajs, device=self.device) * float(pos_ref.iloc[0]['pose_position_z']))

        return torch.stack([px_ref,
                            py_ref,
                            pz_ref], dim=-1) + self.origin

    def vel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            vel_ref = self._df[time_diff == time_diff.min()]
            if vel_ref.shape[0] > 1:
                vel_ref = vel_ref.iloc[0, :]
        else:
            vel_ref = self._df.tail(1)        

        vx_ref = (torch.ones(self.num_trajs, device=self.device) * float(vel_ref.iloc[0]['velocity_linear_x']))
        vy_ref = (torch.ones(self.num_trajs, device=self.device) * float(vel_ref.iloc[0]['velocity_linear_y']))
        vz_ref = (torch.ones(self.num_trajs, device=self.device) * float(vel_ref.iloc[0]['velocity_linear_z']))
            
        return torch.stack([vx_ref,
                            vy_ref,
                            vz_ref], dim=-1)

    def acc(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            acc_ref = self._df[time_diff == time_diff.min()]
            if acc_ref.shape[0] > 1:
                acc_ref = acc_ref.iloc[0, :]
        else:
            acc_ref = self._df.tail(1)          

        ax_ref = (torch.ones(self.num_trajs, device=self.device) * float(acc_ref.iloc[0]['acceleration_linear_x']))
        ay_ref = (torch.ones(self.num_trajs, device=self.device) * float(acc_ref.iloc[0]['acceleration_linear_y']))
        az_ref = (torch.ones(self.num_trajs, device=self.device) * float(acc_ref.iloc[0]['acceleration_linear_z']))
            
        return torch.stack([ax_ref,
                            ay_ref,
                            az_ref], dim=-1)

    def jerk(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            jrk_ref = self._df[time_diff == time_diff.min()]
            if jrk_ref.shape[0] > 1:
                jrk_ref = jrk_ref.iloc[0, :]
        else:
            jrk_ref = self._df.tail(1)           

        jx_ref = (torch.ones(self.num_trajs, device=self.device) * float(jrk_ref.iloc[0]['jerk_linear_x']))
        jy_ref = (torch.ones(self.num_trajs, device=self.device) * float(jrk_ref.iloc[0]['jerk_linear_y']))
        jz_ref = (torch.ones(self.num_trajs, device=self.device) * float(jrk_ref.iloc[0]['jerk_linear_z']))
            
        return torch.stack([jx_ref,
                            jy_ref,
                            jz_ref], dim=-1)

    def snap(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            snp_ref = self._df[time_diff == time_diff.min()]
            if snp_ref.shape[0] > 1:
                snp_ref = snp_ref.iloc[0, :]
        else:
            snp_ref = self._df.tail(1)                     

        sx_ref = (torch.ones(self.num_trajs, device=self.device) * float(snp_ref.iloc[0]['snap_linear_x']))
        sy_ref = (torch.ones(self.num_trajs, device=self.device) * float(snp_ref.iloc[0]['snap_linear_y']))
        sz_ref = (torch.ones(self.num_trajs, device=self.device) * float(snp_ref.iloc[0]['snap_linear_z']))
            
        return torch.stack([sx_ref,
                            sy_ref,
                            sz_ref], dim=-1)

    def yaw(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            yaw_ref = self._df[time_diff == time_diff.min()]
            if yaw_ref.shape[0] > 1:
                yaw_ref = yaw_ref.iloc[0, :]
        else:
            yaw_ref = self._df.tail(1)             

        hd_ref = (torch.ones(self.num_trajs, device=self.device) * float(yaw_ref.iloc[0]['heading']))
            
        return torch.stack([hd_ref], dim=-1)

    def yawvel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, torch.Tensor):
            assert t.shape[0] == self.num_trajs
        else:
            t = torch.ones(self.num_trajs) * t
        if t <= self._time_period:
            time_diff = abs(self._df['time_from_start'] - float(t[0]))
            yrt_ref = self._df[time_diff == time_diff.min()]
            if yrt_ref.shape[0] > 1:
                yrt_ref = yrt_ref.iloc[0, :]
        else:
            yrt_ref = self._df.tail(1)

        yr_ref = (torch.ones(self.num_trajs, device=self.device) * float(yrt_ref.iloc[0]['heading_rate']))
            
        return torch.stack([yr_ref], dim=-1)

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch_control.utils.rot_utils as ru
    ref = TrajFile(num_trajs=1)
    t = torch.stack([torch.arange(0, 10, 0.1) for _ in range(1)], dim=0)
    pos = ref.pos(t).squeeze(0).cpu().numpy()
    vel = ref.vel(t).squeeze(0).cpu().numpy()
    acc = ref.acc(t).squeeze(0).cpu().numpy()
    jerk = ref.jerk(t).squeeze(0).cpu().numpy()
    snap = ref.snap(t).squeeze(0).cpu().numpy()
    quat = ref.quat(t).squeeze(0)
    ang = ru.quat2euler(quat)
    ang = ang.cpu().numpy()
    angvel = ref.angvel(t).squeeze(0).cpu().numpy()
    yaw = ref.yaw(t).squeeze(0) #.cpu().numpy()
    yawvel = ref.yawvel(t).squeeze(0) #.cpu().numpy()
    yawacc = ref.yawacc(t).squeeze(0) #.cpu().numpy()
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
    plt.savefig('circle.png')
    
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
    plt.savefig('circle_xyz.png')
    
    
    