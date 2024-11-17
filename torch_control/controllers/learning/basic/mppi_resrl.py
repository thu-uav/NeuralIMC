import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_control.controllers.base import Controller, register_controller
from torch_control.tasks.base import GymEnv
from torch_control.utils import rot_utils as ru
from torch_control.controllers.classic import MPPI

from .ppo import PPO

def quat_distance(q1, q2):
    '''
    q1: tensor, (N, 4)
    q2: tensor, (N, 4)
    output: tensor, (N,)
    distance = 1 - <q1,q2>^2
    '''
    return 1 - torch.einsum('...i, ...i -> ...', q1, q2)**2

class MPPIResRL(PPO):
    def __init__(self, cfg, envs: GymEnv, eval_env_dict: GymEnv, device: str):
        super().__init__(cfg, envs, eval_env_dict, device)

        self.mppi = MPPI(cfg.mppi, envs, eval_env_dict, device)

        self.minimize_residual_err = cfg.minimize_residual_err
        self.k_pos = cfg.k_pos
        self.k_vel = cfg.k_vel
        self.k_quat = cfg.k_quat
        self.k_raw = cfg.k_raw

    @torch.no_grad()
    def __call__(self, obs_dict):
        rl_action = super().__call__(obs_dict['env_obs'])
        mppi_action = self.mppi.__call__(obs_dict)
        return rl_action + mppi_action

    def init_aux_vars(self, num_envs):
        self.mppi.init_aux_vars(num_envs)

    def setup_params(self):
        super().setup_params()
        self.mppi.setup_params()

    def mppi_infer(self):
        def reference_fn(time):
            ref = self.envs.get_task_reference(time)
            pos = ref.get('position')
            vel = ref.get('linear_velocity', torch.zeros_like(pos))
            quat = ref.get('orientation', torch.zeros(pos.shape[:-1] + (4,)))
            ref_vec = torch.cat([pos, vel, quat], dim=-1)
            return ref_vec
        obs_dict = {
            'state': self.envs.robots.state.clone(),
            'reference_fn': reference_fn,
            'time': self.envs.time
        }
        return self.mppi.infer(obs_dict)

    def _tune_params(self):
        self.make_buffer()
        # warmup buffer
        obs = self.envs.reset()
        self.buffer.reset()
        self.buffer.observations[0].copy_(obs)

        if self.train_predictive_model:
            self.traj_buffer.init(self.envs.get_state_obs(body_frame=False))
        
        self.actor.reset_noise(self.envs.num_envs)

        # start training
        for epoch in range(self.num_epochs):
            # clear gradients of envs
            self.envs.clear_grad()
            
            # collect data
            self.timer.start_timer("rollout")

            train_info = {}

            rollout_info = {
                'pos_err': [],
                'vel_err': [],
                'quat_err': [],
                'reward_minres': [],
                'raw_rewards': []
            }
            
            with torch.no_grad():
                for step in range(self.rollout_length):
                    obs = self.buffer.observations[step]
                    raw_actions, actions, action_log_probs, values = self.policy_infer(obs)
                    mppi_actions, expected_state = self.mppi_infer()
                    combined_actions = actions + mppi_actions
                    next_obs, raw_rewards, dones, info = self.envs.step(combined_actions)
                    achieved_state = self.envs.robots.state
                    achieved_state_vec = torch.cat([achieved_state.pos, achieved_state.vel, achieved_state.quat], dim=-1)

                    if self.minimize_residual_err:
                        e_pos, e_vel, e_quat = expected_state.split([3, 3, 4], dim=-1)
                        a_pos, a_vel, a_quat = achieved_state_vec.split([3, 3, 4], dim=-1)

                        pos_err = torch.norm(e_pos - a_pos, dim=-1)
                        reward_pos = torch.exp(-pos_err)
                        vel_err = torch.norm(e_vel - a_vel, dim=-1)
                        reward_vel = torch.exp(-vel_err)
                        quat_err = ru.euler_distance(e_quat, a_quat)
                        reward_quat = torch.exp(-quat_err)

                        rollout_info['pos_err'].append(pos_err.mean().cpu().numpy())
                        rollout_info['vel_err'].append(vel_err.mean().cpu().numpy())
                        rollout_info['quat_err'].append(quat_err.mean().cpu().numpy())

                        reward_minres = self.k_pos * reward_pos + \
                                        self.k_vel * reward_vel + \
                                        self.k_quat * reward_quat

                        rollout_info['reward_minres'] = reward_minres.mean().cpu().numpy()
                        rollout_info['raw_rewards'] = raw_rewards.mean().cpu().numpy()

                        raw_rewards = self.k_raw * raw_rewards + reward_minres

                    if self.train_predictive_model:
                        self.traj_buffer.add(self.envs.get_state_obs(body_frame=False), 
                                             actions, info["progress"], dones)

                    if 'episode' in info:
                        import pdb; pdb.set_trace()
                        train_info.update(info['episode'])
                    
                    rewards = raw_rewards * self.reward_scaling
                    # import pdb; pdb.set_trace()
                    
                    # handle truncation for episodic envs
                    if len(dones.nonzero(as_tuple=True)[0]) > 0:
                        truncation_mask = dones.float() * info["truncation"].float()
                        next_values = self.critic_infer(info["obs_before_reset"], denorm=True)
                        rewards = rewards + self.gamma * next_values * truncation_mask

                    self.buffer.add(
                        next_obs, raw_actions, values, rewards, dones, action_log_probs)
                    
                self.timer.end_timer("rollout")

                if self.minimize_residual_err:
                    rollout_info = {"rollout_" + k: np.mean(np.array(v)) 
                                    for k, v in rollout_info.items()}
                else:
                    rollout_info = {}

                # compute returns and advantages
                bootstrap_obs = self.buffer.observations[-1]
                bootstrap_value = self.critic_infer(bootstrap_obs, denorm=False)

                self.buffer.compute_returns_and_advantage(
                    bootstrap_value, 
                    value_normalizer=self.rms_value if self.normalize_value else None)

            # update PPO controller parameters
            ppo_info = self.train_op(self.buffer)
            train_info.update(ppo_info)
            train_info.update(rollout_info)
            train_info["time/rollout_fps"] = self.rollout_length * self.envs.num_envs / \
                self.timer.get_time("rollout", mean=True)
            
            if self.train_predictive_model:
                train_info.update(self.update_predictor())

            num_steps = (epoch + 1) * self.rollout_length * self.envs.num_envs

            if epoch % self.log_interval == 0:
                self.log(train_info, num_steps, f"Training [Epoch {epoch}] [Step {num_steps/1e6:.3f} M]", "train")
                self.timer.report()

            if epoch % self.eval_interval == 0 and self.eval_interval > 0:
                eval_info = self.eval()
                self.log(eval_info, num_steps, f"Evaluating [Epoch {epoch}]", "eval")

            if epoch % self.save_interval == 0 and self.save_interval > 0:
                self.save_params()

            self.buffer.after_update()

    def export_deploy_funcs(self):
        super().export_deploy_funcs()
        self.mppi.export_deploy_funcs()

register_controller("mppi_resrl", MPPIResRL)