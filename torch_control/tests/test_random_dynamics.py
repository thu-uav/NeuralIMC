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
from torch_control.predictive_models import get_predictive_model
from torch_control.tasks import GymEnv
from torch_control.utils.wandb_utils import init_wandb
from torch_control.utils.common import set_all_seed
from torch_control.utils import rot_utils as ru

import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../torch_control/configs")

def shorten_string(s):
    # Splitting the string into arguments
    args = re.split(r",(?![^\[\]]*\])", s)

    # Keeping the most important part of each argument and handling list structures correctly
    shortened_args = []
    for arg in args:
        # Extract the key name from the last segment before "=" and retain the entire value
        key = arg.split("=")[0].split(".")[-1]
        value = arg.split("=")[1]
        shortened_arg = f"{key}={value}"
        shortened_args.append(shortened_arg)

    # Joining the arguments back into a string
    return ",".join(shortened_args)

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_quadrotor")
def main(cfg):
    overrided_args = HydraConfig.get().job.override_dirname

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.early_termination = False
    cfg.task.action.offset_g = True

    set_all_seed(cfg.seed)

    envs = GymEnv.REGISTRY[cfg.task.name](cfg, cfg.num_envs)

    envs.reset()

    t = 0

    target = []
    reached = []

    for i in range(100):
        # cos with cycle time of 100
        ct = np.sin(t * 0.02 * 2 * np.pi) * 3
        br = np.sin(t * 0.02 * 2 * np.pi) 
        # heavyside function with cycle time of 100
        # ct = 0.0 if t < 50 else 2.0
        # br = 0.0 if t < 50 else 1.0
        action = torch.tensor([[ct, br * 0.5, br * 0.5, br * 0.1]] * cfg.num_envs).to(cfg.device).float()
        envs.step(action)
        target.append(action)
        
        reached_angvel = envs.robots.state.ang_vel
        reached_acc = envs.robots.state.acc - torch.tensor([0., 0., -9.81]).to(cfg.device)
        reached_quat = envs.robots.state.quat

        reached_thrust = ru.inv_rotate_vector(reached_acc, reached_quat, mode='quat') - torch.tensor([0., 0., 9.81]).to(cfg.device)
        reached_thrust = reached_thrust[..., 2:]

        reached.append(torch.cat([reached_thrust, reached_angvel], dim=-1))

        t += 1
    # import pdb; pdb.set_trace()
    error = torch.stack(reached) - torch.stack(target)
    print("Mean error:", error.abs().mean(0).mean(0))
    print("Std error:", error.abs().mean(0).std(0))
    
    target_np = torch.stack(target).cpu().detach().numpy()
    reached_np = torch.stack(reached).cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    for i in range(1024):
        plt.plot([t * 0.02 for t in range(100)], reached_np[:, i, 0], label="reached")
    plt.plot([t * 0.02 for t in range(100)], target_np[:, 0, 0], 'b--', linewidth=3, label="target")
    # plt.legend()
    plt.subplot(4, 1, 2)
    for i in range(1024):
        plt.plot([t * 0.02 for t in range(100)], reached_np[:, i, 1], label="reached")
    plt.plot([t * 0.02 for t in range(100)], target_np[:, 0, 1], 'b--', linewidth=3, label="target")
    # plt.legend()
    plt.subplot(4, 1, 3)
    for i in range(1024):
        plt.plot([t * 0.02 for t in range(100)], reached_np[:, i, 2], label="reached")
    plt.plot([t * 0.02 for t in range(100)], target_np[:, 0, 2], 'b--', linewidth=3, label="target")
    # plt.legend()
    plt.subplot(4, 1, 4)
    for i in range(1024):
        plt.plot([t * 0.02 for t in range(100)], reached_np[:, i, 3], label="reached")
    plt.plot([t * 0.02 for t in range(100)], target_np[:, 0, 3], 'b--', linewidth=3, label="target")
        # plt.legend()
    plt.tight_layout()
    plt.savefig("error.png")


if __name__ == "__main__":
    main()