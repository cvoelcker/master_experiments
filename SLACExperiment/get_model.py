import os
import torch

from torch_runner.experiment_setup import get_run_path, find_next_run_number

from spatial_monet.spatial_monet import MaskedAIR
from models.monet_stove import MONetStove 
from models.dynamics import Dynamics
from models.slac_stove import SLACAgent, SACAgent
from models.rl_nets import GraphHead, SimpleGraphHead, GraphPolicyNet, GraphQNet, LinearQNet, LinearPolicyNet, ImageQNet
from models.slac_model import SLACModel
from util import buffer


def get_latent_model(config, model_class, load_run, run_name, run_number):
    if config.RL.use_stove:
        monet_config = config.MODULE.MONET
        stove_config = config.MODULE.STOVE
        dynamics_config = config.MODULE.DYNAMICS
        monet = MaskedAIR(**monet_config._asdict())
        dynamics = Dynamics(dynamics_config, monet_config, stove_config)
        stove = MONetStove(stove_config, dynamics, monet)
        if load_run or run_name == 'cheat':
            print('Loading old model')
            path = get_run_path(config.EXPERIMENT.experiment_dir, run_name, run_number)
<<<<<<< Updated upstream
            if run_name == 'cheat':
                path = '../STOVETraining/experiments/visdom-test/run_029/'
            if run_name == 'atari_test':
                path = 'atari-stove-save/run_000/'
=======
            path = '../STOVETraining/experiments/visdom-test/run_029/'
>>>>>>> Stashed changes
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
    else:
        return SLACModel(config.DATA.shape, config.MODULE.DYNAMICS.action_space)


def get_slac(config, monet, qnet, qnet2, policy):
    s_config = config.MODULE.SLAC
    slac = SLACAgent(policy, qnet, qnet2, monet, **s_config._asdict())
    return slac


def get_sac(config, monet, qnet, qnet2, policy):
    s_config = config.MODULE.SLAC
    slac = SACAgent(policy, qnet, qnet2, monet, **s_config._asdict())
    return slac


def get_rl_models(config, load_run, run_name, run_number):
    if config.EXPERIMENT.algorithm == 'slac':
        if config.RL.use_stove:
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
        else:
            q_net_1 = LinearQNet(config.MODULE.SLAC.slac_latent_dim, action_space = config.MODULE.DYNAMICS.action_space)
            q_net_2 = LinearQNet(config.MODULE.SLAC.slac_latent_dim, action_space = config.MODULE.DYNAMICS.action_space)
            policy_net = LinearPolicyNet(config.MODULE.SLAC.slac_latent_dim, action_space = config.MODULE.DYNAMICS.action_space)
        return policy_net.cuda(), q_net_1.cuda(), q_net_2.cuda()
    elif config.EXPERIMENT.algorithm == 'sac':
        q_net_1 = ImageQNet(config.DATA.shape, action_space = config.MODULE.DYNAMICS.action_space)
        q_net_2 = ImageQNet(config.DATA.shape, action_space = config.MODULE.DYNAMICS.action_space)
        return None, q_net_1.cuda(), q_net_2.cuda()
