import logging
import os

import wandb
from omegaconf import OmegaConf, DictConfig


def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    run_dir = os.path.join("outputs", wandb_cfg.group, wandb_cfg.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=wandb_cfg.run_name,
        dir=run_dir,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )

    if wandb_cfg.run_id is not None and wandb_cfg.run_path is None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()

    run = wandb.init(**kwargs)

    if wandb_cfg.mode != "disabled":
        cfg.run_dir = run.dir
    else:
        from hydra import utils
        from hydra.core.hydra_config import HydraConfig
        cfg.run_dir = os.path.join(utils.get_original_cwd(), HydraConfig.get().run.dir)
    
    if wandb_cfg.run_id is not None and run.resumed:  
        # because wandb sweep forces resumed=True
        logging.info(f"Trying to resume run {wandb_cfg.run_id}")
        cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
        run.config.update(cfg_dict)
        checkpoint_name = run.summary["checkpoint"]
        if checkpoint_name is not None:
            logging.info(f"Restore checkpoint {checkpoint_name}")
            wandb.restore(checkpoint_name)
    elif wandb_cfg.run_path is not None:
        logging.info(f"Trying to start new run from {wandb_cfg.run_path}")
        api = wandb.Api()
        run.config = api.run(wandb_cfg.run_path).config
        run.config["old_config"] = run.config.copy()
        cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
        run.config.update(cfg_dict)
        checkpoint_name = run.summary.get("checkpoint")
        if checkpoint_name is not None:
            logging.info(f"Restore checkpoint {checkpoint_name}")
            wandb.restore(checkpoint_name, run_path=wandb_cfg.run_path)
    else:
        cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
        run.config.update(cfg_dict)

    return run
