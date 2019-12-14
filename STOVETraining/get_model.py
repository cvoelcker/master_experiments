import os
import torch

from torch_runner.experiment_setup import get_run_path, find_next_run_number

from spatial_monet.spatial_monet import MaskedAIR
from models.monet_stove import MONetStove
from models.dynamics import Dynamics
from models.slac_stove import SLACAgent, GraphHead, GraphQNet, GraphPolicyNet


def get_model(config, model_class, load_run, run_name, run_number):
    monet_config = config.MODULE.MONET
    stove_config = config.MODULE.STOVE
    dynamics_config = config.MODULE.DYNAMICS
    monet = MaskedAIR(**monet_config._asdict())
    dynamics = Dynamics(dynamics_config, monet_config, stove_config)
    stove = MONetStove(stove_config, dynamics, monet)
    if load_run:
        print('Loading old model')
        path = get_run_path(config.EXPERIMENT.experiment_dir, run_name, run_number)
        path = os.path.join(path, 'checkpoints')
        if not os.path.exists(path):
            print('Found no model checkpoints')
            sys.exit(1)
        try:
            checkpoint_number = config.EXPERIMENT.checkpoint_number
        except AttributeError as e:
            print('Did not specify checkpoint number, using last available')
            checkpoint_number = find_next_run_number(path) - 1
        print(f'Loading checkpoint number {checkpoint_number}')
        path = os.path.join(path, 'model_state_{:07d}.save'.format(checkpoint_number))
        model_state_dict = torch.load(path)
        stove.load_state_dict(model_state_dict)
    return stove


def get_rl_model(config, load_run, run_name, run_number):
    print('SLAC reload is not implemented yet!')

    # path for joint graph head for policy and qnet (joint actor critic nets for better rep learning)
    graph_head = GraphHead(config.SLAC.GRAPH)
    q_net = GraphQNet(graph_head)

    qnet = models
    pass


def get_slac(config, monet, qnet, policy):
    pass
