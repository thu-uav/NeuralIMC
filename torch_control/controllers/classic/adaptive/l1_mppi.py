import os
from typing import Any, Callable, Dict, Tuple

import torch

from torch_control.controllers.classic.basic.mppi import MPPI, register_controller
from torch_control.controllers.l1ac import L1AC, L1AC_batch
from torch_control.tasks.base import GymEnv
from torch_control.utils import rot_utils as ru


class L1_MPPI(MPPI):
    
    def __init__(self, cfg, envs: GymEnv, eval_envs_dict: Dict[str, GymEnv], device: str):
        super().__init__(cfg, envs, eval_envs_dict, device)
        self.cfg = cfg
        # self.l1ac = L1AC(cfg.l1ac, device)
        # self.l1ac.use_l1 = True # force to use L1AC

        self.force_t = None

    def init_aux_vars(self, num_envs: int):
        super().init_aux_vars(num_envs)
        self.force_t = torch.zeros((num_envs, 3)).float().to(self.device)
        self.l1ac = L1AC_batch(num_envs, self.cfg.l1ac, self.device)

    def __call__(self, pid_obs: Dict[str, Any]) -> Tuple[torch.Tensor]:
        state = pid_obs.get('state') # QuadrotorState
        time = pid_obs.get('time') # torch.Tensor
        reference_fn = pid_obs.get('reference_fn') # Dict
        
        pos, vel, quat = state.pos, state.vel, state.quat
        state_vec = torch.cat([pos, vel, quat], dim=-1)

        if self.force_t is None:
            self.force_t = torch.zeros_like(vel) # (num_envs, 3)

        external_acc = pid_obs.get('wind_vec') #self.l1ac.adaptation_fn(vel, self.force_t, self.ctrl_dt)
                        
        action, self.eval_action_mean, self.eval_u, self.eval_angvel = self.compute_action(state_vec, time, reference_fn, 
                                                            self.eval_action_mean, self.eval_u, self.eval_angvel, external_acc)
        action[..., 1:] = ru.inv_rotate_vector(action[..., 1:], quat, mode='quat') # world -> body

        if self.envs.action_cfg.offset_g:
            action[..., 0] -= self.envs.robots.g

        self.force_t = ru.rotate_axis(2, quat, mode='quat') * action[..., 0:1] * self.unit_mass # (num_envs, 3)

        return action

register_controller('l1_mppi', L1_MPPI)