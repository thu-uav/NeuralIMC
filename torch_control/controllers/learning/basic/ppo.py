import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_control.controllers.base import Controller, register_controller
from torch_control.controllers.learning.modules import (MixinExtractor,
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


class PPO(Controller):

    def __init__(self, cfg, envs: GymEnv, eval_envs_dict: GymEnv, device: str):
        super().__init__(cfg, envs, eval_envs_dict, device)

        # PPO parameters
        self.rollout_length = self.cfg.rollout_length
        self.ppo_epochs = self.cfg.ppo_epochs
        self.minibatch_size = self.cfg.minibatch_size
        if (self.envs.num_envs * self.rollout_length) % self.minibatch_size > 0:
            print("[PPO] The number of envs * rollout_length should be divisible by minibatch_size, but " \
            "got {} * {} / {}".format(self.envs.num_envs, self.rollout_length, self.minibatch_size))
        self.num_minibatch = (self.envs.num_envs * self.rollout_length) // self.minibatch_size
        self.clip_ratio = self.cfg.clip_ratio
        self.gae_lambda = self.cfg.gae_lambda
        self.gamma = self.cfg.gamma # discount factor
        
        self.reward_scaling = self.cfg.reward_scaling
        self.max_grad_norm = self.cfg.max_grad_norm
        
        self.normalize_obs = self.cfg.normalize_obs
        self.normalize_value = self.cfg.normalize_value
        self.normalize_advantages = self.cfg.normalize_advantages
        
        self.entropy_coef = self.cfg.entropy_coef
        self.critic_loss_coef = self.cfg.critic_loss_coef
        self.use_clipped_value_loss = self.cfg.use_clipped_value_loss
        self.use_huber_loss = self.cfg.use_huber_loss
        self.huber_delta = self.cfg.huber_delta

        self.total_steps = self.cfg.total_steps
        self.num_epochs = int(self.total_steps // self.rollout_length // self.envs.num_envs)

        self.log_interval = self.cfg.log_interval
        self.save_interval = self.cfg.save_interval
        self.eval_interval = self.cfg.eval_interval

        self.timer.add_timer("rollout")
        
        self.train_predictive_model = False

    def __call__(self, obs: torch.Tensor, reference: Any = None, verbose: bool = False) -> torch.Tensor:
        if self.normalize_obs:
            obs = self.rms_obs.normalize(obs)
        features = self.actor_extractor(obs)
        actions = self.unscale_actions(self.actor(features, deterministic=True))
        if verbose:
            return actions, features
        else:
            return actions

    def init_aux_vars(self, num_envs: int):
        return

    def policy_infer(self, obs: torch.Tensor):
        if self.normalize_obs:
            obs = self.rms_obs.normalize(obs)
        
        actor_features = self.actor_extractor(obs)
        if self.cfg.share_feature_extractor:
            critic_features = actor_features
        else:
            critic_features = self.critic_extractor(obs)

        # compute actions and log probs
        raw_actions, action_log_probs = self.actor.act(actor_features)
        # compute values
        values = self.critic(critic_features)
        
        actions = self.unscale_actions(raw_actions)
        
        return raw_actions, actions, action_log_probs, values
    
    def critic_infer(self, obs: torch.Tensor, denorm: bool = False):
        if self.normalize_obs:
            obs = self.rms_obs.normalize(obs)
        critic_features = self.critic_extractor(obs)
        values = self.critic(critic_features)
        if self.normalize_value and denorm:
            values = self.rms_value.denormalize(values.unsqueeze(-1)).squeeze(-1)
        return values
    
    def unscale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.actor.squash_action:
            return self.action_min + (
                actions + 1.0) * 0.5 * (self.action_max - self.action_min)
        else:
            return actions.clamp(self.action_min, self.action_max)

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
        
        self.optimizer = optim.Adam(parameters, 
                                    betas=self.cfg.betas,
                                    lr=self.cfg.lr, 
                                    eps=self.cfg.lr_eps, 
                                    weight_decay=self.cfg.weight_decay)
        
    def policy_parameters(self):
        if self.cfg.share_feature_extractor:
            return list(self.actor_extractor.parameters()) + list(self.actor.parameters()) + \
                list(self.critic.parameters())
        else:
            return list(self.actor_extractor.parameters()) + list(self.actor.parameters()) + \
                list(self.critic_extractor.parameters()) + list(self.critic.parameters())

    def actor_parameters(self):
        return list(self.actor_extractor.parameters()) + list(self.actor.parameters())

    def critic_parameters(self):
        if self.cfg.share_feature_extractor:
            return list(self.critic.parameters())
        else:
            return list(self.critic_extractor.parameters()) + list(self.critic.parameters())

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
            
            with torch.no_grad():
                for step in range(self.rollout_length):
                    obs = self.buffer.observations[step]
                    raw_actions, actions, action_log_probs, values = self.policy_infer(obs)
                    next_obs, raw_rewards, dones, info = self.envs.step(actions)
                    
                    if self.train_predictive_model:
                        self.traj_buffer.add(self.envs.get_state_obs(body_frame=False), 
                                             actions, info["progress"], dones)

                    if 'episode' in info:
                        train_info.update(info['episode'])
                    
                    rewards = raw_rewards * self.reward_scaling
                    
                    # handle truncation for episodic envs
                    if len(dones.nonzero(as_tuple=True)[0]) > 0:
                        truncation_mask = dones.float() * info["truncation"].float()
                        next_values = self.critic_infer(info["obs_before_reset"], denorm=True)
                        rewards = rewards + self.gamma * next_values * truncation_mask

                    self.buffer.add(
                        next_obs, raw_actions, values, rewards, dones, action_log_probs)
                    
                self.timer.end_timer("rollout")
                    
                # compute returns and advantages
                bootstrap_obs = self.buffer.observations[-1]
                bootstrap_value = self.critic_infer(bootstrap_obs, denorm=False)

                self.buffer.compute_returns_and_advantage(
                    bootstrap_value, 
                    value_normalizer=self.rms_value if self.normalize_value else None)

            # update PPO controller parameters
            ppo_info = self.train_op(self.buffer)
            train_info.update(ppo_info)
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
            
    def make_feature_extrator(self, feature_extractor_cfg, override_cfg, obs_size_dict, activation_fn=None):
        obs_dict = {}
        for k in obs_size_dict:
            kwargs = feature_extractor_cfg.get(k)
            if override_cfg is not None:
                kwargs = override_cfg.get(k, kwargs)
            if isinstance(obs_size_dict[k], int):
                obs_length = obs_size_dict[k]
            elif isinstance(obs_size_dict[k], tuple):
                obs_length = np.prod(obs_size_dict[k], dtype=np.int32)
                unit_dim = int(obs_size_dict[k][-1])
                kwargs.update({"unit_dim": unit_dim})
            else:
                raise NotImplementedError(f"Obs size {obs_size_dict[k]} not supported.")
            obs_dict[k] = (obs_length, kwargs)
        return MixinExtractor(obs_dict)

    def make_actor(self, actor_cfg, input_dim: int, action_dim: int):
        return MLPPolicy(actor_cfg, input_dim, action_dim)
    
    def make_critic(self, critic_cfg, input_dim: int):
        return ValueNet(critic_cfg, input_dim)
    
    def make_policy(self,):
        feature_extractor = self.make_feature_extrator(
            self.cfg.actor.feature_extractor, 
            self.cfg.actor.feature_extractor_override, 
            self.envs.obs_size_dict,
            activation_fn=getattr(nn, self.cfg.actor.activation))

        parameters = list(feature_extractor.parameters())

        if self.cfg.share_feature_extractor:
            self.actor_extractor = feature_extractor
            self.critic_extractor = feature_extractor
        else:
            self.actor_extractor = feature_extractor
            self.critic_extractor = self.make_feature_extrator(
                self.cfg.critic.feature_extractor, 
                self.cfg.critic.feature_extractor_override, 
                self.envs.obs_size_dict,
                activation_fn=getattr(nn, self.cfg.critic.activation))
            parameters.extend(list(self.critic_extractor.parameters()))
            
        actor_input_dim = int(self.actor_extractor.feature_dim)
        critic_input_dim = int(self.critic_extractor.feature_dim)
        
        self.actor = self.make_actor(self.cfg.actor, actor_input_dim, self.action_dim, )
        self.critic = self.make_critic(self.cfg.critic, critic_input_dim)
        
        parameters.extend(list(self.actor.parameters()) +
                          list(self.critic.parameters()))
        
        self.actor_extractor = self.actor_extractor.to(self.device)
        self.critic_extractor = self.critic_extractor.to(self.device)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        
        return parameters
    
    def make_buffer(self):
        self.buffer = OnpolicyBuffer(self.rollout_length,
                                    obs_space=self.envs.observation_space,
                                    action_space=self.envs.action_space,
                                    device=self.device,
                                    num_envs=self.envs.num_envs,
                                    gae_lambda=self.gae_lambda,
                                    gamma=self.gamma,
                                    normalize_advantages=self.normalize_advantages,)

    def train_op(self, buffer):
        self.set_train()
        
        train_info = {}
        if self.normalize_obs:
            self.rms_obs.update(buffer.observations)
        if self.normalize_value:
            self.rms_value.update(buffer.returns.unsqueeze(-1))

        for _ in range(self.ppo_epochs):
            sampler = buffer.get_sampler(self.num_minibatch)
            for minibatch in sampler:
                update_info = self.ppo_update(minibatch)
                for k, v in update_info.items():
                    if k not in train_info:
                        train_info[k] = 0
                    train_info[k] += v

        num_updates = self.ppo_epochs * self.num_minibatch
        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info["explained_variance"] = buffer.get_explained_var().item()
        train_info["rollout/advantages_mean"] = buffer.advantages.mean().item()
        train_info["rollout/rewards_mean"] = buffer.rewards.mean().item()
        train_info["rollout/returns_mean"] = buffer.returns.mean().item()
        train_info["rollout/values_mean"] = buffer.values.mean().item()
        train_info["rollout/raw_actions_mean"] = buffer.actions.mean().item()
        
        self.set_eval()

        return train_info
    
    def ppo_update(self, minibatch) -> Dict[str, float]:
        self.actor.reset_noise(minibatch.observations.shape[0])
        
        obs = minibatch.observations
        if self.normalize_obs:
            obs = self.rms_obs.normalize(obs)
            
        actor_features = self.actor_extractor(obs)
        if self.cfg.share_feature_extractor:
            critic_features = actor_features
        else:
            critic_features = self.critic_extractor(obs)
            
        action_log_probs, dist_entropy = self.actor.evaluate_actions(actor_features, minibatch.actions)
        values = self.critic(critic_features)

        actor_loss, ratio = self.compute_actor_loss(action_log_probs, 
                                                    minibatch.action_log_probs, 
                                                    minibatch.advantages)

        critic_loss = self.compute_critic_loss(values, 
                                               minibatch.values, 
                                               minibatch.returns)

        clip_fraction = (torch.abs(ratio - 1) > self.clip_ratio).float()
        with torch.no_grad():
            log_ratio = action_log_probs - minibatch.action_log_probs
            approx_kl_div = (torch.exp(log_ratio) - 1) - log_ratio

        # update actor_critic
        self.optimizer.zero_grad()

        (
            actor_loss \
            - dist_entropy * self.entropy_coef \
            + critic_loss * self.critic_loss_coef
        ).backward()
        
        actor_grad_norm = get_grad_norm(self.actor.parameters())
        critic_grad_norm = get_grad_norm(self.critic.parameters())
        actor_extractor_grad_norm = get_grad_norm(self.actor_extractor.parameters())
        critic_extractor_grad_norm = get_grad_norm(self.critic_extractor.parameters())
        policy_grad_norm = get_grad_norm(self.policy_parameters())

        nan_alarm(actor_grad_norm)
        nan_alarm(critic_grad_norm)
        nan_alarm(actor_extractor_grad_norm)
        nan_alarm(critic_extractor_grad_norm)

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_parameters(), self.max_grad_norm)
            
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "dist_entropy": dist_entropy.item(),
            "ratio": ratio.mean().item(),
            "clip_fraction": clip_fraction.mean().item(),
            "approx_kl_div": approx_kl_div.mean().item(),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "actor_extractor_grad_norm": actor_extractor_grad_norm,
            "critic_extractor_grad_norm": critic_extractor_grad_norm,
            "policy_grad_norm": policy_grad_norm,
            "log_std": self.actor.log_std.exp().mean().item(),
        }

    def compute_actor_loss(self, action_log_probs, old_action_log_probs, advantages):
        ratio = torch.exp(action_log_probs - old_action_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 
                            1.0 - self.clip_ratio, 
                            1.0 + self.clip_ratio) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        nan_alarm(actor_loss)

        return actor_loss, ratio
    
    def compute_critic_loss(self, values, old_values, returns):
        values_clipped = old_values + (
            values - old_values).clamp(-self.clip_ratio, self.clip_ratio)

        if self.normalize_value:
            returns = self.rms_value.normalize(returns.unsqueeze(-1)).squeeze(-1)

        error_clipped = returns - values_clipped
        error = returns - values

        if self.use_huber_loss:
            value_losses = huber_loss(error, delta=self.huber_delta)
            value_losses_clipped = huber_loss(error_clipped, delta=self.huber_delta)
        else:
            value_losses = error.pow(2) * 0.5
            value_losses_clipped = error_clipped.pow(2) * 0.5

        if self.use_clipped_value_loss:
            critic_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            critic_loss = value_losses.mean()

        return critic_loss
    
    def save_params(self, suffix: str = None):
        suffix = "" if suffix is None else f"_{suffix}"
        if self.normalize_obs:
            torch.save(self.rms_obs.state_dict(), os.path.join(self.save_dir, f"rms_obs{suffix}.pth"))
        if self.normalize_value:
            torch.save(self.rms_value.state_dict(), os.path.join(self.save_dir, f"rms_value{suffix}.pth"))
        torch.save(self.actor_extractor.state_dict(), os.path.join(self.save_dir, f"actor_extractor{suffix}.pth"))
        if not self.cfg.share_feature_extractor:
            torch.save(self.critic_extractor.state_dict(), os.path.join(self.save_dir, f"critic_extractor{suffix}.pth"))
        torch.save(self.actor.state_dict(), os.path.join(self.save_dir, f"actor{suffix}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(self.save_dir, f"critic{suffix}.pth"))
        
        if hasattr(self, 'predictive_model') and isinstance(self.predictive_model, nn.Module):
            torch.save(self.predictive_model.state_dict(), os.path.join(self.save_dir, f"predictor{suffix}.pth"))
    
    def set_train(self):
        self.actor_extractor.train()
        if not self.cfg.share_feature_extractor:
            self.critic_extractor.train()
        self.actor.train()
        self.critic.train()

    def set_eval(self):
        self.actor_extractor.eval()
        if not self.cfg.share_feature_extractor:
            self.critic_extractor.eval()
        self.actor.eval()
        self.critic.eval()
        
    def export_deploy_funcs(self,):
        """
        Export the functions for deployment.
        """
        self.set_eval()
        self.time = None
        self.is_warmup = True
        
        def init_task(odom_dict: Dict, task_dict: Dict = None):
            state_vec = torch.cat([odom_dict["position"], 
                                   odom_dict["linear_velocity"],
                                   odom_dict["orientation"],
                                   odom_dict["angular_velocity"]], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            
            if task_dict is not None:
                task_vec = torch.cat([task_dict["position"],
                                      task_dict["orientation"]], dim=-1)
                task_vec = task_vec.unsqueeze(0).float().to(self.device)
            else:
                task_vec = None
            
            self.envs.reset(state_vec=state_vec, task_vec=task_vec)
            for eval_name in self.eval_envs_dict:
                self.eval_envs_dict[eval_name].reset(state_vec=state_vec, task_vec=task_vec)
            
            self.last_state = self.envs.state.clone()
            self.last_cmd = self.envs.cmd.clone()

        def infer(odom_dict: Dict, time: float = None, last_cmd: torch.Tensor = None) -> torch.Tensor:
            if last_cmd is None:
                last_cmd = self.last_cmd
            last_cmd = last_cmd.to(self.device)

            state_vec = torch.cat([odom_dict["position"],
                                   odom_dict["linear_velocity"],
                                   odom_dict["orientation"],
                                   odom_dict["angular_velocity"]], dim=-1)
            state_vec = state_vec.unsqueeze(0).float().to(self.device)
            state = QuadrotorState.construct_from_vec(state_vec).to(self.device)

            if time is None:
                time_vec = self.envs.time
            else:
                time_vec = torch.tensor([time]).float().to(self.device)

            obs = self.envs.get_obs(state=state,
                                    last_state=self.last_state,
                                    time=time_vec,
                                    cmd=last_cmd,
                                    enable_noise=False)
            ref = self.envs.get_task_reference(time_vec)
            self.envs.progress += 1
            self.envs.time += self.envs.robots.control_dt
            self.last_state = state.clone()

            action, latent = self.__call__(obs, verbose=True)
            self.last_cmd = action.clone()
            action = self.envs.process_action(action)
            
            if hasattr(self.envs.robots, "wind_l1ac"):
                l1ac_log = self.envs.robots.wind_l1ac.d_hat.squeeze(0)
            else:
                l1ac_log = torch.zeros(3)

            # add logs
            logging_data = {
                'time': time_vec.squeeze(0).tolist(),
                'pos_est': state.pos.squeeze(0).tolist(),
                'pos_ref': ref['position'].squeeze(0).tolist(),
                'vel_est': state.vel.squeeze(0).tolist(),
                'vel_ref': ref.get('linear_velocity', torch.zeros(1, 3)).squeeze(0).tolist(),
                'att_est': state.quat.squeeze(0).tolist(),
                'att_ref': ref['orientation'].squeeze(0).tolist(),
                'omg_est': state.ang_vel.squeeze(0).tolist(),
                'omg_ref': action.squeeze(0)[1:].tolist(),
                'thr_ref': action.squeeze(0)[:1].tolist(),
                'obs': obs.squeeze(0).tolist(),
                'latent': latent.squeeze(0).tolist(),
                'l1ac': l1ac_log.tolist()
            }

            logging_info = {
                'time': time,
                'state': odom_dict,
                'ref': {k: v.squeeze(0) for k, v in ref.items()},
                'l1ac': l1ac_log,
            }

            return action.squeeze(0), logging_data, logging_info
        
        return init_task, infer
    
    def set_predictive_model(self, pm_cfg, predictive_model, no_random_steps: int = 0):
        self.pm_cfg = pm_cfg
        self.train_predictive_model = self.pm_cfg.train
        self.no_random_steps = no_random_steps
        
        if self.train_predictive_model:
            self.traj_buffer = TrajectoryBuffer(
                buffer_size=self.pm_cfg.buffer_size, 
                episode_length=self.envs.max_episode_length,
                state_dim=self.envs.state_size,
                action_dim=self.envs.action_size,
                num_envs=self.envs.num_envs,
                device=self.device,
                use_priority=self.pm_cfg.use_priority,)
                
        self.predictive_model = predictive_model
    
    def update_predictor(self):
        if self.no_random_steps < 0:
            self.train_predictive_model = False
            return {}
        elif self.no_random_steps > 0 and self.envs.elapsed_steps > self.no_random_steps:
            if self.train_predictive_model:
                self.train_predictive_model = False
            return {}

        self.predictive_model.train()

        infos = {
            'pred/pred_state_loss': [],
            'pred/grad_norm': [],
            'pred/priority': [],
        }
        for _ in range(self.pm_cfg.num_iteration):
            sampler = self.traj_buffer.get_sampler(self.predictive_model.train_horizon,
                                                   self.pm_cfg.num_minibatch,
                                                   self.pm_cfg.minibatch_size,
                                                   device=self.device)
            for minibatch in sampler:
                train_info = self.predictive_model.update_step(minibatch.state, minibatch.action)
                infos['pred/pred_state_loss'].append(train_info['pred_state_loss'].mean().item())
                infos['pred/grad_norm'].append(train_info['grad_norm'].item())
                infos['pred/priority'].append(minibatch.priority.mean().item())
                for state_i in range(self.predictive_model.state_dim):
                    info_name = f'pred/pred_state_loss/{state_i}'
                    if info_name not in infos:
                        infos[info_name] = []
                    infos[info_name].append(train_info['pred_state_loss'][..., state_i].mean().item())
                
        for k in infos:
            infos[k] = np.mean(infos[k])

        self.predictive_model.eval()
            
        with torch.no_grad():
            self.envs.update_predictive_model(self.predictive_model)
            for env_key in self.eval_envs_dict.keys():
                self.eval_envs_dict[env_key].update_predictive_model(self.predictive_model)
        
        return infos

register_controller("ppo", PPO)
