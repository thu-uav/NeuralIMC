import abc
from functools import partial
from typing import Dict, List, Union

import torch
import torch.nn as nn

from torch_control.utils.rot_utils import quat2matrix, matrix2quat
from torch_control.utils.rot_tools import compute_rotation_matrix_from_ortho6d

def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p

def rot_transform(src_rot, tgt_rot, rot):
    if src_rot == 'matrix' and tgt_rot == 'quat': # 9 -> 4
        return matrix2quat(rot.reshape(-1, 3, 3))
    elif src_rot == 'quat' and tgt_rot == 'quat': # 4 -> 4
        return rot
    elif src_rot == 'matrix' and tgt_rot == '6dof': # 9 -> 6
        return rot.reshape(-1, 3, 3)[:, :, :2].reshape(-1, 6)
    elif src_rot == 'quat' and tgt_rot == '6dof': # 4 -> 6
        return quat2matrix(rot).reshape(-1, 3, 3)[:, :, :2].reshape(-1, 6)
    else:
        raise ValueError(f"Unsupported rotation transformation: {src_rot} -> {tgt_rot}")

def rot_inv_transform(src_rot, tgt_rot, rot, output_quat=False):
    if output_quat:
        src_rot = 'quat'
    if tgt_rot == 'quat' and src_rot == 'matrix': # 4 -> 9
        return quat2matrix(rot).reshape(-1, 9)
    elif tgt_rot == '6dof' and src_rot == 'matrix': # 9 -> 9
        return compute_rotation_matrix_from_ortho6d(rot).reshape(-1, 9)
    elif tgt_rot == 'quat' and src_rot == 'quat': # 4 -> 4
        return rot
    elif tgt_rot == '6dof' and src_rot == 'quat': # 6 -> 4
        return matrix2quat(compute_rotation_matrix_from_ortho6d(quat2matrix(rot))).reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported rotation transformation: {src_rot} -> {tgt_rot}")

rot_dim_dict = {
    'matrix': 9,
    'quat': 4,
    '6dof': 6,
}

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, action_dim, rot_mode='quat', device='cuda'):
        super().__init__()

        self.raw_state_dim = state_dim
        self.action_dim = action_dim
        self.tgt_rot = rot_mode

        assert rot_mode in ['quat', '6dof'], f"Unsupported rot_mode: {rot_mode}"
        
        if self.raw_state_dim == 18: # pos, vel, rot_matrix, angvel
            self.src_rot = 'matrix'
            self.angvel = True
        elif self.raw_state_dim == 15: # pos, vel, rot_matrix
            self.src_rot = 'matrix'
            self.angvel = False
        elif self.raw_state_dim == 13: # pos, vel, rot_quat, angvel
            self.src_rot = 'quat'
            self.angvel = True
        elif self.raw_state_dim == 10: # pos, vel, rot_quat
            self.src_rot = 'quat'
            self.angvel = False
        else:
            raise ValueError(f"Unsupported state_dim for 6dof rotation: {self.state_dim}")

        self.rot_src2tgt = partial(rot_transform, self.src_rot, self.tgt_rot)
        self.rot_tgt2src = partial(rot_inv_transform, self.src_rot, self.tgt_rot)

        self.src_rot_dim = rot_dim_dict[self.src_rot]
        self.tgt_rot_dim = rot_dim_dict[self.tgt_rot]
        self.state_dim = self.raw_state_dim - rot_dim_dict[self.src_rot] + rot_dim_dict[self.tgt_rot]
        
        self.device = device
        self.aux_vars = {}

    def forward(self, x, u, output_quat=False):
        return self.predict(x, u, output_quat)
    
    def predict(self, x, u, output_quat=False):
        raise NotImplementedError
    
    @abc.abstractmethod
    def unroll(self, x0, us):
        raise NotImplementedError
    
    @abc.abstractmethod
    def update_step(self, state, action):
        raise NotImplementedError
    
    def preprocess_state(self, x):
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]) # flatten batch dims

        if self.angvel:
            pos, vel, rot, angvel = x.split([3, 3, self.src_rot_dim, 3], dim=1)
        else:
            pos, vel, rot = x.split([3, 3, self.src_rot_dim], dim=1)

        tgt_rot = self.rot_src2tgt(rot)

        if self.angvel:
            x = torch.cat([pos, vel, tgt_rot, angvel], dim=1)
        else:
            x = torch.cat([pos, vel, tgt_rot], dim=1)

        x = x.reshape(*batch_shape, -1) # restore batch dims

        return x
    
    def postprocess_state(self, x, output_quat=False):
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if self.angvel:
            pos, vel, rot, angvel = x.split([3, 3, self.tgt_rot_dim, 3], dim=1)
        else:
            pos, vel, rot = x.split([3, 3, self.tgt_rot_dim], dim=1)

        src_rot = self.rot_tgt2src(rot, output_quat)

        if self.angvel:
            x = torch.cat([pos, vel, src_rot, angvel], dim=1)
        else:
            x = torch.cat([pos, vel, src_rot], dim=1)

        x = x.reshape(*batch_shape, -1)
        return x
    

class Repeat(TrajectoryModel):
    def __init__(self, state_dim, action_dim, rot_mode='quat', device='cuda'):
        super().__init__(state_dim, action_dim, rot_mode, device)

    def __repr__(self):
        return "RepeatDynamics"

    def reset_aux_vars(self, idx: Union[int, torch.Tensor]):
        pass

    def predict(self, x, u, output_quat=False):
        x = self.preprocess_state(x)
        xtp1 = self.postprocess_state(x, output_quat)

        return xtp1