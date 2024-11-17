import copy
import os
import random
import re

import hydra
import numpy as np
import setproctitle
import torch
import torch_control
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch_control.controllers import Controller
from torch_control.tasks import GymEnv
from torch_control.utils.wandb_utils import init_wandb
from torch_control.utils import rot_utils as ru

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs")

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("[Info] Setting all random seeds to {}".format(seed))
    
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

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    overrided_args = HydraConfig.get().job.override_dirname
    if overrided_args:
        cfg.wandb.run_name += '-' + shorten_string(overrided_args)
    if cfg.exp_name is not None:
        cfg.wandb.run_name = cfg.exp_name

    if cfg.task.name == "track":
        cfg.controller.feature_extractor.task_obs.net = 'simple_tcn'

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    set_all_seed(cfg.seed)
    setproctitle.setproctitle(cfg.wandb.run_name)
    
    cfg.noise.obs_std = 0.
    cfg.latency_obs = 0.
    cfg.latency_action = 0.
    cfg.auto_reset = False
    
    cfg.wind.enable = True
    cfg.wind.random_walk = True
    cfg.wind.use_l1ac = True
    

    print("-----" * 5, ' Training Environment ', "-----" * 5)
    envs = GymEnv.REGISTRY[cfg.task.name](cfg, cfg.num_envs)
    envs.reset()
    
    all_errors = []
    all_l1wind = []
    all_wind = []
    
    for _ in range(100):
        actions = torch.stack([
            torch.from_numpy(envs.action_space.sample()) for _ in range(cfg.num_envs)]).float().to(envs.device)
        
        obs, _, _, _ = envs.step(actions)
        estimated_wind = obs[:, -3:]
        true_wind = envs.robots.wind_vec
        
        error = estimated_wind - true_wind
        
        print("Error: ", error.abs().mean(0))
        
        all_errors.append(error)
        all_l1wind.append(estimated_wind)
        all_wind.append(true_wind)
        
    all_errors = torch.stack(all_errors)
    all_l1wind = torch.stack(all_l1wind)
    all_wind = torch.stack(all_wind)
    
    max_error_idx = all_errors.abs().mean(dim=0).argmax(dim=0)
    max_error_idx = max_error_idx[0]
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 5))
    plt.subplot(3, 1, 1)
    plt.plot(all_l1wind[:, max_error_idx, 0].cpu().numpy(), label='l1wind_x')
    plt.plot(all_wind[:, max_error_idx, 0].cpu().numpy(), label='wind_x')
    plt.title('wind_x')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(all_l1wind[:, max_error_idx, 1].cpu().numpy(), label='l1wind_y')
    plt.plot(all_wind[:, max_error_idx, 1].cpu().numpy(), label='wind_y')
    plt.title('wind_y')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(all_l1wind[:, max_error_idx, 2].cpu().numpy(), label='l1wind_z')
    plt.plot(all_wind[:, max_error_idx, 2].cpu().numpy(), label='wind_z')
    plt.title('wind_z')
    plt.legend()
    plt.tight_layout()
    plt.savefig('l1ac_test.png')
    
if __name__ == "__main__":
    main()