import os
from typing import Any, Callable, Dict, Tuple

import torch

from torch_control.controllers.base import Controller, register_controller
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.tasks.base import GymEnv
from torch_control.utils import rot_utils as ru
from torch_control.utils.integrators import so3_quat_integral, exp_euler2quat, quat_multiply

POS_SLICE = slice(0, 3)
VEL_SLICE = slice(3, 6)
QUAT_SLICE = slice(6, 10)

def quat_distance(q1, q2):
    '''
    q1: tensor, (N, 4)
    q2: tensor, (N, 4)
    output: tensor, (N,)
    distance = 1 - <q1,q2>^2
    '''
    return 1 - torch.einsum('...i, ...i -> ...', q1, q2)**2

class MPPI(Controller):
    
    def __init__(self, cfg, envs: GymEnv, eval_envs_dict: Dict[str, GymEnv], device: str):
        super().__init__(cfg, envs, eval_envs_dict, device)

        self.unit_mass = 1.0
        self.g = 9.81
        self.ctrl_dt = self.envs.robots.control_dt
        
        self.horizon = self.cfg.H
        self.num_samples = self.cfg.N
        self.num_envs = self.envs.num_envs
        
        self.thrust_hover = self.unit_mass * self.g
        self.time_horizon = torch.arange(0, self.horizon * self.ctrl_dt, self.ctrl_dt).float().to(self.device) # (horizon,)
        
        sample_std = self.cfg.sample_std
        self.sample_std = torch.tensor((
            sample_std[0] * self.thrust_hover,
            sample_std[1],
            sample_std[2],
            sample_std[3],
        )).float().to(self.device)
        
        self.sim_k = 0.4
        self.time_step = 0
        
        self.action_min = torch.as_tensor(self.envs.action_space.low).to(self.device)
        self.action_max = torch.as_tensor(self.envs.action_space.high).to(self.device)

        if self.envs.action_cfg.offset_g:
            self.action_min[0] += self.envs.robots.g
            self.action_max[0] += self.envs.robots.g

        self.lam = self.cfg.lam
        self.alpha_p_xy = self.cfg.alpha_p_xy
        self.alpha_p_z = self.cfg.alpha_p_z
        self.alpha_w = self.cfg.alpha_w
        self.alpha_a = self.cfg.alpha_a
        self.alpha_R = self.cfg.alpha_R
        self.alpha_v_xy = self.cfg.alpha_v_xy
        self.alpha_v_z = self.cfg.alpha_v_z
        self.alpha_z = self.cfg.alpha_z
        self.alpha_yaw = self.cfg.alpha_yaw

    def init_aux_vars(self, num_envs: int):
        self.eval_u = torch.zeros((num_envs, self.num_samples, self.horizon)).float().to(self.device)
        self.eval_angvel = torch.zeros((num_envs, self.num_samples, self.horizon, 3)).float().to(self.device)
        self.eval_action_mean = torch.zeros(num_envs, self.horizon, 4).float().to(self.device)
        self.eval_action_mean[..., 0] = self.thrust_hover

    def setup_params(self):
        self.u = torch.zeros((self.num_envs, self.num_samples, self.horizon)).float().to(self.device)
        self.angvel = torch.zeros((self.num_envs, self.num_samples, self.horizon, 3)).float().to(self.device)
        self.action_mean = torch.zeros(self.num_envs, self.horizon, 4).float().to(self.device)
        self.action_mean[..., 0] = self.thrust_hover

    def reset_params(self, env_ids):
        self.u[env_ids, ...] = 0
        self.angvel[env_ids, ...] = 0
        self.action_mean[env_ids, ...] = 0
        self.action_mean[env_ids, ..., 0] = self.thrust_hover

    def __call__(self, pid_obs: Dict[str, Any]) -> Tuple[torch.Tensor]:
        state = pid_obs.get('state') # QuadrotorState
        time = pid_obs.get('time') # torch.Tensor
        reference_fn = pid_obs.get('reference_fn') # Dict
        
        pos, vel, quat = state.pos, state.vel, state.quat
        state_vec = torch.cat([pos, vel, quat], dim=-1)
        
        action, self.eval_action_mean, self.eval_u, self.eval_angvel = self.compute_action(state_vec, time, reference_fn, 
                                                            self.eval_action_mean, self.eval_u, self.eval_angvel)
        action[..., 1:] = ru.inv_rotate_vector(action[..., 1:], quat, mode='quat') # world -> body

        if self.envs.action_cfg.offset_g:
            action[..., 0] -= self.envs.robots.g

        return action
    
    def infer(self, pid_obs: Dict[str, Any]) -> Tuple[torch.Tensor]:
        state = pid_obs.get('state') # QuadrotorState
        time = pid_obs.get('time') # torch.Tensor
        reference_fn = pid_obs.get('reference_fn') # Dict
        
        pos, vel, quat = state.pos, state.vel, state.quat
        state_vec = torch.cat([pos, vel, quat], dim=-1)
        
        action, self.action_mean, self.u, self.angvel = self.compute_action(
            state_vec, time, reference_fn, self.action_mean, self.u, self.angvel)
        action[..., 1:] = ru.inv_rotate_vector(action[..., 1:], quat, mode='quat') # world -> body

        next_state = self.compute_expected_state(state_vec, action)

        if self.envs.action_cfg.offset_g:
            action[..., 0] -= self.envs.robots.g

        return action, next_state
    
    def compute_expected_state(self, state_vec, action):
        pos, vel, quat = torch.split(state_vec, [3, 3, 4], dim=-1)
        e3 = torch.tensor([0, 0, 1]).float().to(self.device)
        dt = self.ctrl_dt

        u = action[..., 0] # (num_envs,)
        angvel = action[..., 1:] # (num_envs, 3)

        next_state = torch.zeros((self.num_envs, 1, 10)).float().to(self.device)

        dang = (angvel * dt).unsqueeze(1) # (num_envs, 1, 3)
        next_state = self.quat_integral_loop(quat, dang, next_state, 1)
        next_state = next_state.squeeze(1)

        z_vec = ru.rotate_axis(2, next_state[..., QUAT_SLICE], mode='quat')
        acc = u.unsqueeze(-1) * z_vec - self.g * e3.view(1, 3)

        next_state[..., VEL_SLICE] = vel + acc * dt
        next_state[..., POS_SLICE] = pos + next_state[..., VEL_SLICE] * dt

        # import pdb; pdb.set_trace()

        return next_state
    
    def compute_action(self, state_vec, time, reference_fn, action_mean, u, angvel,
                       external_acc=None):
        action_mean = action_mean.clone()
        u = u.clone()
        angvel = angvel.clone()

        action_mean_old = action_mean.clone()
        action_mean[..., :-1, :] = action_mean_old[..., 1:, :]

        states, actions, u, angvel = self.rollout_states(state_vec, action_mean, u, angvel, external_acc)

        ref_time = time.unsqueeze(-1) + self.time_horizon # (num_envs, horizon)
        ref_states = []
        for n_t in range(self.horizon):
            ref_states_at_t = reference_fn(ref_time[:, n_t])
            ref_states.append(ref_states_at_t)
        ref_states = torch.stack(ref_states, dim=1).unsqueeze(1) # (num_envs, 1, horizon, state_dim)
        # ref_states = ref_states.unsqueeze(1) # (num_envs, 1, horizon, state_dim)

        pos_err = states[..., POS_SLICE] - ref_states[..., POS_SLICE] # (num_envs, num_samples, horizon, 3)
        vel_err = states[..., VEL_SLICE] - ref_states[..., VEL_SLICE] # (num_envs, num_samples, horizon, 3)
        cost = self.alpha_p_xy * torch.sum(torch.linalg.norm(pos_err[..., :2], dim=-1), dim=-1) + \
            self.alpha_p_z * torch.sum(torch.abs(pos_err[..., 2]), dim=-1) + \
            self.alpha_R * torch.sum(quat_distance(states[..., QUAT_SLICE], ref_states[..., QUAT_SLICE]), dim=-1) + \
            self.alpha_v_xy * torch.sum(torch.linalg.norm(vel_err[..., :2], dim=-1), dim=-1) + \
            self.alpha_v_z * torch.sum(torch.abs(vel_err[..., 2]), dim=-1)
        
        cost = cost * self.ctrl_dt # (num_envs, num_samples)
        cost = cost - torch.min(cost, dim=-1, keepdim=True)[0] # (num_envs, num_samples)
        weight = torch.softmax(-cost / self.lam, dim=-1) # (num_envs, num_samples)

        action_mean = torch.sum(actions * weight[..., None, None], dim=1) # (num_envs, horizon, 4)

        action_final = action_mean[..., 0, :] # (num_envs, horizon)
        action_final[..., 0] = action_final[..., 0] / self.unit_mass # (num_envs, horizon)

        return action_final, action_mean, u, angvel

    def quat_integral_loop(self, quat, dang, states, horizon):
        for h in range(horizon):
            states[..., h, QUAT_SLICE] = so3_quat_integral(quat, dang[..., h, :], dang_in_body=False)
            quat = states[..., h, QUAT_SLICE] # (num_envs, num_samples, 4)

        return states
    
    def qmultiply_loop(self, quat, rotquat, states, horizon):
        for h in range(horizon):
            states[..., h, QUAT_SLICE] = quat_multiply(rotquat[..., h, :], quat)
            quat = states[..., h, QUAT_SLICE]

        return states
    
    def rollout_states(self, start_state: torch.Tensor, action_mean, u, angvel, external_acc=None):
        state_shape = start_state.shape # (num_envs, state_dim)
        num_envs = state_shape[0]

        pos_0, vel_0, quat_0 = torch.split(start_state.unsqueeze(1).repeat(1, self.num_samples, 1), 
                                           [3, 3, 4], dim=-1) # (num_envs, 1, 3), (num_envs, 1, 3), (num_envs, 1, 4)
        
        e3 = torch.tensor([0, 0, 1]).float().to(self.device) # (3,)
        dt = self.ctrl_dt
        
        actions = torch.normal(
            mean=action_mean.unsqueeze(1).repeat(1, self.num_samples, 1, 1), 
            std=self.sample_std) # (num_envs, num_samples, horizon, 4)
        
        u = u + self.sim_k * (actions[..., 0] / self.unit_mass - u) # (num_envs, num_samples, horizon)
        angvel = angvel + self.sim_k * (actions[..., 1:] - angvel) # (num_envs, num_samples, horizon, 3)

        u = torch.clamp(u, self.action_min[0], self.action_max[0])
        angvel = torch.clamp(angvel, self.action_min[1:], self.action_max[1:])

        states = torch.zeros((num_envs, self.num_samples, self.horizon, state_shape[-1])).float().to(self.device)
        dang = angvel * dt
        states = self.quat_integral_loop(quat_0, dang, states, self.horizon)

        z_vec = ru.rotate_axis(2, states[..., QUAT_SLICE], mode='quat') # (num_envs, num_samples, horizon, 3)
        acc = u.unsqueeze(-1) * z_vec - self.g * e3.view(1, 1, 1, 3)
        if external_acc is not None:
            acc = acc + external_acc[:, None, None, :] # (num_envs, num_samples, horizon, 3)
        
        states[..., VEL_SLICE] = vel_0.unsqueeze(-2) + torch.cumsum(acc * dt, dim=-2) # (num_envs, num_samples, horizon, 3)
        states[..., POS_SLICE] = pos_0.unsqueeze(-2) + torch.cumsum(states[..., VEL_SLICE] * dt, dim=-2) # (num_envs, num_samples, horizon, 3)
        
        return states, actions, u, angvel
    
    def export_deploy_funcs(self):
        self.time = None
        self.init_aux_vars(1)
        
        def init_task(odom_dict: Dict, task_dict: Dict = None):
            state_vec = torch.cat([odom_dict['position'], 
                                   odom_dict['linear_velocity'],
                                   odom_dict['orientation'],
                                   odom_dict['angular_velocity']], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            
            if task_dict is not None:
                task_vec = torch.cat([task_dict['position'],
                                      task_dict['orientation']], dim=-1)
                task_vec = task_vec.unsqueeze(0).float().to(self.device)
            else:
                task_vec = None
            
            self.envs.reset(state_vec=state_vec, task_vec=task_vec)

        def preprocess(odom_dict: Dict, time: float = None) -> torch.Tensor:
            state_vec = torch.cat([odom_dict['position'], 
                                   odom_dict['linear_velocity'],
                                   odom_dict['orientation'],
                                   odom_dict['angular_velocity']], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            state = QuadrotorState.construct_from_vec(state_vec).to(self.device)
            
            time_vec = torch.tensor([time]).float().to(self.device)

            def reference_fn(time):
                ref = self.envs.get_task_reference(time)
                pos = ref.get('position')
                vel = ref.get('linear_velocity', torch.zeros_like(pos))
                quat = ref.get('orientation', torch.zeros(pos.shape[:-1] + (4,)))
                ref_vec = torch.cat([pos, vel, quat], dim=-1)
                return ref_vec

            self.envs.progress += 1
            
            logging_dict = {
                'time': time,
                'state': odom_dict,
                'ref': {k: v.squeeze(0) for k, v in self.envs.get_task_reference(time_vec).items()}
            }
            
            return {'state': state, 'reference_fn': reference_fn, 'time': time_vec}, logging_dict
        
        def controller(obs: Dict) -> torch.Tensor:
            return self.__call__(obs)
        
        def postprocess(actions: torch.Tensor) -> torch.Tensor:
            actions = self.envs.process_action(actions)
            return actions.squeeze(0)
        
        def infer_fn(odom_dict: Dict, time: float = None) -> torch.Tensor:
            obs, logging_info = preprocess(odom_dict, time)
            action = controller(obs)
            action = postprocess(action)
            return action, {}, logging_info
        
        return init_task, infer_fn #preprocess, controller, postprocess
    
    def save_params(self, *args, **kwargs):
        pass
    
    def _tune_params(self):
        pass      
    
    def set_train(self):
        pass
    
    def set_eval(self):
        pass

register_controller('mppi', MPPI)