from typing import Dict

import argparse
import copy
import os
import random
import re
import sys
import tqdm
sys.path.append("..")

import hydra
import numpy as np
import time
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch_control.controllers.base import Controller
from torch_control.controllers.learning.modules import TrajectoryBuffer
from torch_control.dynamics.quadrotor.quadrotor_state import QuadrotorState
from torch_control.sim2real.utils.low_pass_filter import LowPassFilter
from torch_control.tasks.base import GymEnv
from torch_control.utils.visualizer import Visualizer
from torch_control.utils.rot_utils import quat2euler
from torch_control.utils.common import set_all_seed, load_config_from_wandb

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs")

def shorten_string(s):
    # Splitting the string into arguments
    args = re.split(r',(?![^\[\]]*\])', s)

    # Keeping the most important part of each argument and handling list structures correctly
    shortened_args = []
    for arg in args:
        # Extract the key name from the last segment before '=' and retain the entire value
        key = arg.split('=')[0].split('.')[-1]
        value = arg.split('=')[1]
        shortened_arg = f"{key}={value}"
        shortened_args.append(shortened_arg)

    # Joining the arguments back into a string
    return ','.join(shortened_args)

@torch.no_grad()
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="offline")
def main(cfg):
    overrided_args = HydraConfig.get().job.override_dirname
    if overrided_args:
        overrided_args = shorten_string(overrided_args)
        
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    offline_cfg = copy.deepcopy(cfg.offline_collect)
    
    assert cfg.load_from is not None and os.path.exists(cfg.load_from), "Please specify a checkpoint to load from, got {}".format(cfg.load_from)
    
    wandb_cfg = load_config_from_wandb(cfg.load_from)
    cfg = OmegaConf.merge(cfg, wandb_cfg)
    
    if cfg.task.name == "track" and cfg.controller.name == "ppo":
        cfg.controller.feature_extractor.task_obs.net = 'simple_tcn'
    if cfg.quadrotor.name == "train_random":
        cfg.eval.quadrotor = ["eval_random"]
        
    set_all_seed(cfg.seed)
    
    cfg.wandb.mode = 'disabled'
    cfg.wind.use_l1ac = True
    cfg.auto_reset = False
    cfg.random_dynamics = False
    
    env = GymEnv.REGISTRY[cfg.task.name](cfg, offline_cfg.num_envs)
    eval_env = GymEnv.REGISTRY[cfg.task.name](cfg, offline_cfg.num_eval_envs)
    
    controller = Controller.REGISTRY[cfg.controller.algo_name](
        cfg.controller, env, eval_env, device=cfg.device)
    controller.setup_params()
    
    max_episodes = offline_cfg.max_episodes
    num_epochs = max_episodes // offline_cfg.num_envs
    
    if offline_cfg.short_name is None:
        save_dir = offline_cfg.save_dir + f"/{cfg.wandb.run_name}"
    else:
        save_dir = offline_cfg.save_dir + f"/{offline_cfg.short_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    dataset = {
        'states': None,
        'actions': None
    }
    
    controller.set_eval()
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        set_all_seed(cfg.seed + epoch * 100)
        
        env.clear_grad()
        obs = env.reset()
        
        state_list = []
        action_list = []
        
        for i in range(env.max_episode_length):
            raw_actions, actions, action_log_probs, values = controller.policy_infer(obs)
            
            state_list.append(env.get_state_obs(body_frame=False, bodyrate=True).detach().cpu().numpy())
            action_list.append(actions.detach().cpu().numpy())
            
            obs, raw_rewards, dones, info = env.step(actions)
            
        states = np.stack(state_list, axis=1)
        actions = np.stack(action_list, axis=1)
        
        if dataset['states'] is None:
            dataset['states'] = states
            dataset['actions'] = actions
        else:
            dataset['states'] = np.concatenate([dataset['states'], states], axis=0)
            dataset['actions'] = np.concatenate([dataset['actions'], actions], axis=0)

        np.savez(save_dir + '/data.npz', **dataset)
            

if __name__ == "__main__":
    main()
            
            
    
    