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
from torch_control.utils.common import set_all_seed, load_config_from_wandb

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

    if cfg.load_from is not None:
        assert os.path.exists(cfg.load_from), f"Checkpoint {cfg.load_from} does not exist!"
        wandb_cfg = load_config_from_wandb(cfg.load_from)
        seed = cfg.seed
        task = cfg.task
        exp_name = cfg.exp_name
        cfg = OmegaConf.merge(cfg, wandb_cfg)
        cfg.seed = seed
        cfg.task = task
        cfg.exp_name = exp_name

    if cfg.task.name == "track" and cfg.controller.algo_name in ["ppo", "rma", "rma_student"]:
        cfg.controller.feature_extractor.task_obs.net = "simple_tcn"
    if cfg.quadrotor.name == "train_random":
        cfg.eval.quadrotor = ["eval_random"]

    cfg.only_eval=True
    cfg.wandb.mode="disabled"
    cfg.num_envs=1
    cfg.num_eval_envs=1024

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    set_all_seed(cfg.seed)
    setproctitle.setproctitle(cfg.wandb.run_name)

    if overrided_args:
        print("[Info] Raw overrided_args:", overrided_args)

    print("[Info] Quadrotor: {} | Controller: {} | Task: {}".format(
        cfg.quadrotor.name, cfg.controller.name, cfg.task.name))

    ############## Environment ##############
    print("-----" * 5, " Training Environment ", "-----" * 5)
    envs = GymEnv.REGISTRY[cfg.task.name](cfg, cfg.num_envs)

    eval_envs_dict = {}
    print("-----" * 5, f" Eval Environment - default ", "-----" * 5)
    eval_cfg = copy.deepcopy(cfg)
    eval_envs_dict["train"] = GymEnv.REGISTRY[cfg.task.name](
        eval_cfg, eval_cfg.num_eval_envs)

    enable_l1ac = True and cfg.controller.name != 'rma'
    
    if cfg.wind.enable:
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.predictive_model.no_random_steps = -1
        eval_cfg.wind.use_l1ac = enable_l1ac
        eval_envs_dict["train_l1ac"] = GymEnv.REGISTRY[cfg.task.name](
            eval_cfg, eval_cfg.num_eval_envs)

        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.predictive_model.no_random_steps = -1
        eval_cfg.wind.use_l1ac = False
        eval_cfg.wind.enable = False
        eval_envs_dict["train_nowind"] = GymEnv.REGISTRY[cfg.task.name](
            eval_cfg, eval_cfg.num_eval_envs)
        
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.predictive_model.no_random_steps = -1
        eval_cfg.wind.use_l1ac = enable_l1ac
        eval_cfg.wind.enable = False
        eval_envs_dict["train_nowind_l1ac"] = GymEnv.REGISTRY[cfg.task.name](
            eval_cfg, eval_cfg.num_eval_envs)

    eval_quadrotor_list = cfg.eval.quadrotor if cfg.eval.quadrotor else []
    if isinstance(eval_quadrotor_list, str):
        eval_quadrotor_list = [eval_quadrotor_list]

    for quadrotor_name in eval_quadrotor_list:
        print("-----" * 5, f" Eval Environment - {quadrotor_name} ", "-----" * 5)
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.predictive_model.no_random_steps = -1
        eval_cfg.quadrotor = OmegaConf.load(
            os.path.join(CONFIG_PATH, f"quadrotor/{quadrotor_name}.yaml"))
        eval_envs_dict[quadrotor_name] = GymEnv.REGISTRY[eval_cfg.task.name](
            eval_cfg, eval_cfg.num_eval_envs)
        
        if cfg.wind.enable:
            eval_cfg = copy.deepcopy(cfg)
            eval_cfg.predictive_model.no_random_steps = -1
            eval_cfg.wind.use_l1ac = enable_l1ac
            eval_cfg.quadrotor = OmegaConf.load(
                os.path.join(CONFIG_PATH, f"quadrotor/{quadrotor_name}.yaml"))
            eval_envs_dict[quadrotor_name + "_l1ac"] = GymEnv.REGISTRY[eval_cfg.task.name](
                eval_cfg, eval_cfg.num_eval_envs)

            eval_cfg = copy.deepcopy(cfg)
            eval_cfg.predictive_model.no_random_steps = -1
            eval_cfg.wind.use_l1ac = False
            eval_cfg.wind.enable = False
            eval_cfg.quadrotor = OmegaConf.load(
                os.path.join(CONFIG_PATH, f"quadrotor/{quadrotor_name}.yaml"))
            eval_envs_dict[quadrotor_name + "_nowind"] = GymEnv.REGISTRY[eval_cfg.task.name](
                eval_cfg, eval_cfg.num_eval_envs)
            
            eval_cfg = copy.deepcopy(cfg)
            eval_cfg.predictive_model.no_random_steps = -1
            eval_cfg.wind.use_l1ac = enable_l1ac
            eval_cfg.wind.enable = False
            eval_cfg.quadrotor = OmegaConf.load(
                os.path.join(CONFIG_PATH, f"quadrotor/{quadrotor_name}.yaml"))
            eval_envs_dict[quadrotor_name + "_nowind_l1ac"] = GymEnv.REGISTRY[eval_cfg.task.name](
                eval_cfg, eval_cfg.num_eval_envs)

    ############## Controller ##############
    print("-----" * 5, " Controller ", "-----" * 5)
    controller = Controller.REGISTRY[cfg.controller.algo_name](
        cfg.controller, envs, eval_envs_dict, device=cfg.device)

    controller.setup_params()
    # controller.tune_params(only_eval=cfg.only_eval, render=cfg.render, plot=cfg.plot)

    controller.set_eval()
    # eval_info = {}

    f = open(f"sss_{cfg.controller.name}_{cfg.task.trajectory.type}.log", "a")
    f.write(cfg.exp_name + "\n")

    for eval_name, eval_envs in eval_envs_dict.items():
        print("========================================")
        print(f"Evaluating on {eval_name} environments")
        print("----------------------------------------")
        eval_info = controller.eval_once(
            eval_envs, eval_name, render=cfg.render, plot=cfg.plot)

        for key in eval_info.keys():
            if "metrics/error/error_distance" in key:
                f.write(f"{eval_name} {key} {eval_info[key]}\n")
                print(f"{eval_name} {key} {eval_info[key]}\n")

    f.write("\n")
    f.close()
    
    # wandb_run.finish()

if __name__ == "__main__":
    main()