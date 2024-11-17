import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_control.controllers.base import Controller, register_controller
from torch_control.controllers.learning.modules import (AdaptationNetwork,
                                                        MixinExtractor,
                                                        MLPPolicy,
                                                        OnpolicyBuffer,
                                                        TrajectoryBuffer,
                                                        RunningMeanStd,
                                                        ValueNet)
from torch_control.controllers.learning.utils.common import (get_action_dim,
                                                             get_grad_norm,
                                                             get_obs_shape,
                                                             huber_loss)
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.tasks.base import GymEnv
from torch_control.predictive_models import get_predictive_model
from torch_control.utils.common import nan_alarm
from .ppo import PPO


class RMA(PPO):

    def __init__(self, cfg, envs: GymEnv, eval_env_dict: GymEnv, device: str):
        super().__init__(cfg, envs, eval_env_dict, device)

        self.history_horizon = 50
        self.rollout_length = self.cfg.rma_rollout_length
        self.train_iterations = self.cfg.train_iterations
        self.lr = 0.001

    @torch.no_grad()
    def __call__(self, obs):
        if self.normalize_obs:
            obs = self.rms_obs.normalize(obs)

        actor_features = self.actor_extractor(obs)

        # update history
        self.eval_history = torch.cat((self.eval_history[:, :, 1:], obs[:, self.history_obs_slice].unsqueeze(-1)), dim=2)

        adapt_features = self.adapt_extractor(self.eval_history)

        common_features = actor_features[:, :self.common_feature_size]
        param_features = actor_features[:, self.common_feature_size:]

        new_actor_features = torch.cat((common_features, adapt_features), dim=1)

        raw_actions, _ = self.actor.act(new_actor_features)
        actions = self.unscale_actions(raw_actions)

        return actions #, common_features, adapt_features, param_features

    def init_aux_vars(self, num_envs):
        self.eval_history = torch.zeros((num_envs, self.history_obs_size, self.history_horizon)).to(self.device)

    def infer(self, obs, history):
        if self.normalize_obs:
            obs = self.rms_obs.normalize(obs)

        with torch.no_grad():
            actor_features = self.actor_extractor(obs)

        adapt_features = self.adapt_extractor(history)

        common_features = actor_features[:, :self.common_feature_size]
        param_features = actor_features[:, self.common_feature_size:]

        new_actor_features = torch.cat((common_features, adapt_features), dim=1)

        with torch.no_grad():
            raw_actions, _ = self.actor.act(new_actor_features)
            actions = self.unscale_actions(raw_actions)

        return actions, common_features, adapt_features, param_features

    def rollout_adaptive_policy(self):
        all_e_pred = None
        all_e_gt = None

        history = torch.zeros((self.envs.num_envs, self.history_obs_size, self.history_horizon)).to(self.device)

        obs = self.envs.reset()

        for i in range(self.rollout_length):
            history = torch.cat((history[:, :, 1:], obs[:, self.history_obs_slice].unsqueeze(-1)), dim=2)
            actions, _, e_pred, e_gt = self.infer(obs, history)
            obs, rewards, dones, infos = self.envs.step(actions)

            if all_e_gt is None:
                all_e_gt = e_gt
                all_e_pred = e_pred
            else:
                all_e_gt = torch.cat((all_e_gt, e_gt), dim=0)
                all_e_pred = torch.cat((all_e_pred, e_pred), dim=0)

        return all_e_pred, all_e_gt

    def setup_params(self):
        self.obs_dim = np.prod(get_obs_shape(self.envs.observation_space), dtype=np.int32)
        self.action_dim = self.envs.action_size
        
        self.action_min = torch.as_tensor(self.envs.action_space.low).to(self.device)
        self.action_max = torch.as_tensor(self.envs.action_space.high).to(self.device)
        
        parameters = self.make_policy()
        
        if self.normalize_obs:
            self.rms_obs = RunningMeanStd(shape=self.obs_dim, device=self.device)
            
        if self.normalize_value:
            self.rms_value = RunningMeanStd((1,), device=self.device)
        
        if self.cfg.ckpt_dir is not None:
            print("[Info] Loading checkpoint from {}".format(self.cfg.ckpt_dir))
            if self.cfg.share_feature_extractor:
                self.actor_extractor.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "actor_extractor_best_train.pth")))
                self.critic_extractor = self.actor_extractor
            else:
                self.actor_extractor.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "actor_extractor_best_train.pth")))
                self.critic_extractor.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "critic_extractor_best_train.pth")))
            
            self.actor.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "actor_best_train.pth")))
            
            if self.normalize_obs:
                self.rms_obs.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "rms_obs_best_train.pth")))
                
            try:
                self.critic.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "critic_best_train.pth")))
            except:
                print(f"[WARNING] No critic checkpoint found. Using random critic.")

            if self.normalize_value:
                try:
                    self.rms_value.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "rms_value_best_train.pth")))
                except:
                    print(f"[WARNING] No value normalization checkpoint found. Using random value normalization.")
        else:
            print("[Info] Initializing parameters")
            self.actor.init_params()
            self.critic.init_params()
        
        # self.optimizer = optim.Adam(parameters, 
        #                             betas=self.cfg.betas,
        #                             lr=self.cfg.lr, 
        #                             eps=self.cfg.lr_eps, 
        #                             weight_decay=self.cfg.weight_decay)
        
        self.history_obs_slice = slice(np.prod(self.envs.task_obs_size), 
                                       self.envs.obs_size - self.envs.ext_obs_size - self.envs.int_obs_size)
        self.history_obs_size = self.history_obs_slice.stop - self.history_obs_slice.start

        if self.envs.int_obs_size > 0:
            int_feature_size = self.cfg.actor.feature_extractor.int_obs.output_dim
        else:
            int_feature_size = 0
        if self.envs.ext_obs_size > 0:
            ext_feature_size = self.envs.ext_obs_size
        else:
            ext_feature_size = 0

        self.adapt_feature_size = ext_feature_size + int_feature_size
        self.common_feature_size = self.actor_extractor.feature_dim - self.adapt_feature_size

        self.adapt_extractor = AdaptationNetwork(self.history_obs_size, 
                                                 self.adapt_feature_size, complex=False).to(self.device)

        if self.cfg.ckpt_dir is not None:
            print("[Info] Loading checkpoint from {}".format(self.cfg.ckpt_dir))
            self.adapt_extractor.load_state_dict(torch.load(os.path.join(self.cfg.ckpt_dir, "adapt_net_best_train.pth")))

        self.optimizer = optim.Adam(self.adapt_extractor.parameters(), lr=self.lr)

    def _tune_params(self):
        running_loss = 0.0
        count = 0

        steps_per_rollout = self.rollout_length * self.envs.num_envs
        best_loss = 999.

        for i in range(self.train_iterations):
            self.optimizer.zero_grad()

            all_e_pred, all_e_gt = self.rollout_adaptive_policy()

            loss = F.mse_loss(all_e_pred, all_e_gt)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.detach().cpu().item()
            count += 1

            train_info = {'loss': loss.detach().cpu().item()}

            if i % 10 == 0:
                train_info['running_loss'] = running_loss / count
                best_loss = min(best_loss, train_info['running_loss'])

                if best_loss == train_info['running_loss']:
                    self.save_params("best")

                running_loss = 0.0
                count = 0

                eval_info = self.eval()
                self.log(eval_info, (i + 1) * steps_per_rollout, "Evaluation", "eval")

                self.save_params()

            self.log(train_info, (i + 1) * steps_per_rollout, f"Train Iteration {i}", "train")

            self.save_params("final")

    def export_deploy_funcs(self):
        raise NotImplementedError
        return super().export_deploy_funcs()

    def save_params(self, suffix: str = None):
        super().save_params(suffix)
        suffix = "" if suffix is None else f"_{suffix}"
        torch.save(self.adapt_extractor.state_dict(), os.path.join(self.save_dir, f"adapt_net{suffix}.pth"))

    def set_train(self):
        self.adapt_extractor.train()

    def set_eval(self):
        self.adapt_extractor.eval()

register_controller("rma_student", RMA)