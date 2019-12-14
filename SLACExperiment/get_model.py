import os
import torch

from torch_runner.experiment_setup import get_run_path, find_next_run_number

from spatial_monet.spatial_monet import MaskedAIR
from models.monet_stove import MONetStove 
from models.dynamics import Dynamics
from models.slac_stove import GraphHead, SimpleGraphHead, GraphPolicyNet, GraphQNet, SLACAgent
from util import buffer


def get_latent_model(config, model_class, load_run, run_name, run_number):
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


def get_slac(config, monet, qnet, qnet2, policy):
    s_config = config.MODULE.SLAC
    slac = SLACAgent(policy, qnet, qnet2, monet, **s_config._asdict())
    return slac


def get_rl_models(config, load_run, run_name, run_number):
    combine = config.RL.combined_nets
    q_config = config.MODULE.SLAC.QNET
    p_config = config.MODULE.SLAC.POLICY
    g_config = config.MODULE.SLAC.POLICY

    if combine:
        graph_head1 = graph_head2 = graph_head3 = SimpleGraphHead(num_slots = config.MODULE.MONET.num_slots, cl = config.MODULE.STOVE.cl)
    else:
        graph_head1 = SimpleGraphHead(num_slots = config.MODULE.MONET.num_slots, cl = config.MODULE.STOVE.cl)
        graph_head2 = SimpleGraphHead(num_slots = config.MODULE.MONET.num_slots, cl = config.MODULE.STOVE.cl)
        graph_head3 = SimpleGraphHead(num_slots = config.MODULE.MONET.num_slots, cl = config.MODULE.STOVE.cl)
    q_net_1 = GraphQNet(graph_head1, action_space = config.MODULE.DYNAMICS.action_space)
    q_net_2 = GraphQNet(graph_head2, action_space = config.MODULE.DYNAMICS.action_space)
    policy_net = GraphPolicyNet(graph_head3, action_space = config.MODULE.DYNAMICS.action_space)
    return policy_net, q_net_1, q_net_2