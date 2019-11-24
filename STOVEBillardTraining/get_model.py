import os
import torch

from torch_runner.experiment_setup import get_run_path, find_next_run_number

from spatial_monet.spatial_monet import MaskedAIR
from monet_stove import MONetStove
from dynamics import Dynamics


def get_model(config, model_class):
    monet_config = config.MODULE.MONET
    stove_config = config.MODULE.STOVE
    dynamics_config = config.MODULE.DYNAMICS
    monet = MaskedAIR(**monet_config._asdict())
    dynamics = Dynamics(dynamics_config, monet_config, stove_config)
    stove = MONetStove(stove_config, dynamics, monet)
    ex = config.EXPERIMENT
    print(ex.load_run)
    if ex.load_run:
        path = get_run_path(ex.experiment_dir, ex.run_name, ex.run_number)
        path = os.path.join(path, 'checkpoints')
        if not os.path.exists(path):
            print('Found no model checkpoints')
            sys.exit(1)
        try:
            checkpoint_number = ex.checkpoint_number
        except AttributeError as e:
            print('Did not specify checkpoint number, using last available')
            checkpoint_number = find_next_run_number(path) - 1
        path = os.path.join(path, 'model_state_{:07d}.save'.format(checkpoint_number))
        model_state_dict = torch.load(path)
        stove.load_state_dict(model_state_dict)
    if hasattr(ex, 'load_monet') and ex.load_monet:
        path = get_run_path(ex.monet.experiment_dir, ex.monet.run_name, ex.monet.run_number)
        path = os.path.join(path, 'checkpoints')
        if not os.path.exists(path):
            print('Found no model checkpoints')
            sys.exit(1)
        try:
            checkpoint_number = ex.checkpoint_number
        except AttributeError as e:
            print('Did not specify checkpoint number, using last available')
            checkpoint_number = find_next_run_number(path) - 1
        path = os.path.join(path, 'model_{:03d}'.format(checkpoint_number))
        model_state_dict = torch.load(path)
        monet.load_state_dict(model_state_dict)
    return stove
