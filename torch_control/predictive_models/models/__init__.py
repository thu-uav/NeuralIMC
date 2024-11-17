from .prior.rigid_body import RigidBody
from .rec_mlp import RecMLPDynamics
from .transformer.transformer import TransformerDynamics
from .base import Repeat

def get_predictive_model(model_arch, cfg, state_dim, action_dim, max_ep_len, rot_mode, device):
    if model_arch == 'rec_mlp':
        return RecMLPDynamics(cfg, state_dim, action_dim, rot_mode, device)
    elif model_arch == 'transformer':
        return TransformerDynamics(cfg, state_dim, action_dim, max_ep_len, rot_mode, device)
    elif model_arch == 'repeat':
        return Repeat(state_dim, action_dim, rot_mode, device)
    #elif model_arch == 'diffusion':
    #    return DiffusionDynamics(cfg, state_dim, action_dim, rot_mode, device)
    else:
        raise NotImplementedError(f'Unknown model_arch {model_arch}')
