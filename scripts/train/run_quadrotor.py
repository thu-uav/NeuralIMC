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

    if cfg.task.name == "track" and cfg.controller.algo_name in ["ppo", "rma", "rma_student"]:
        cfg.controller.feature_extractor.task_obs.net = "simple_tcn"
    if cfg.quadrotor.name == "train_random":
        cfg.eval.quadrotor = ["eval_random"]
    if cfg.task.name == "track":
        cfg.wandb.run_name += f"_{cfg.task.trajectory.type}"
    if overrided_args:
        cfg.wandb.run_name += "-" + shorten_string(overrided_args)
    if cfg.exp_name is not None:
        cfg.wandb.run_name = cfg.exp_name

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    set_all_seed(cfg.seed)
    setproctitle.setproctitle(cfg.wandb.run_name)
    
    wandb_run = init_wandb(cfg)
    cfg.controller.save_dir = os.path.join(cfg.run_dir, "checkpoints")
    if not os.path.exists(cfg.controller.save_dir):
        os.makedirs(cfg.controller.save_dir)

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
        eval_cfg.wind.enable = False
        eval_envs_dict["train_nowind"] = GymEnv.REGISTRY[cfg.task.name](
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
            eval_cfg.wind.use_l1ac = True
            eval_cfg.wind.enable = False
            eval_cfg.quadrotor = OmegaConf.load(
                os.path.join(CONFIG_PATH, f"quadrotor/{quadrotor_name}.yaml"))
            eval_envs_dict[quadrotor_name + "_nowind_l1ac"] = GymEnv.REGISTRY[eval_cfg.task.name](
                eval_cfg, eval_cfg.num_eval_envs)


    ############## Controller ##############
    print("-----" * 5, " Controller ", "-----" * 5)
    controller = Controller.REGISTRY[cfg.controller.algo_name](
        cfg.controller, envs, eval_envs_dict, device=cfg.device)

    ############## Predictive Model ##############
    if cfg.predictive_model.enable:
        if cfg.predictive_model.ckpt is not None:
            assert os.path.exists(cfg.predictive_model.ckpt), f"ckpt not found: {cfg.predictive_model.ckpt}"
            ckpt_cfg = OmegaConf.load(os.path.join(cfg.predictive_model.ckpt, "model_cfg.yaml"))
            model_arch = ckpt_cfg.arch
            model_cfg = ckpt_cfg.get(model_arch)
            model_ckpt = torch.load(os.path.join(cfg.predictive_model.ckpt, 'best_model.pth'), map_location=cfg.device)
        else:
            model_arch = cfg.predictive_model.arch
            model_cfg = cfg.predictive_model.get(cfg.predictive_model.arch)
            model_ckpt = None

        no_random_steps = cfg.predictive_model.no_random_steps
        
        predictive_model = get_predictive_model(model_arch=model_arch,
                                                cfg=model_cfg,
                                                state_dim=envs.state_size,
                                                action_dim=envs.action_size,
                                                max_ep_len=envs.max_episode_length,
                                                rot_mode=cfg.predictive_model.rot_mode,
                                                device=cfg.device,)
        
        if model_ckpt is not None:
            predictive_model.load_state_dict(model_ckpt)
            print(f"Loaded predictive model from {cfg.predictive_model.ckpt}")
        
        envs.set_predictive_model(predictive_model, no_random_steps)
        for env_key in eval_envs_dict.keys():
            eval_envs_dict[env_key].set_predictive_model(predictive_model)
        controller.set_predictive_model(cfg.predictive_model, predictive_model, no_random_steps)

    controller.setup_params()
    controller.tune_params(only_eval=cfg.only_eval, render=cfg.render, plot=cfg.plot)

    wandb_run.finish()

if __name__ == "__main__":
    main()