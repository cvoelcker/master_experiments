from sys import argv

import torch
import gym
from config_parser.config_parser import ConfigGenerator

from src.experiments.monet_a2c_training import RLACTrainer
from src.experiments.monet_ppo_training import RLPPOTrainer
from src.experiments.handlers import PrintHandler, TensorboardHandler
from src.experiments.arg_parser import parse_args, parse_conf_file, \
    cast_conf_split
from src.rl_algorithms.networks import ActorCriticImageNet, \
    CNNBase, FCNetwork, ActorCritic, ObjectDetectionNetwork
from src.rl_algorithms.agents import ACAgent
from src.util.gym_util import load_rl_data
from src.util.torch_util import get_optimizer


def build_ppo_experiment(conf):
    """
    Builds an rl experiment runner from the provided config attribute
    :param conf: a parsed config
    :return: a ready trainer
    """
    
    # read all values from config
    embedding_size = conf.INPUT_SPEC.embedding_size
    optimizer = get_optimizer(conf.TRAIN_SPEC.optimizer)
    step_size = conf.TRAIN_SPEC.step_size
    step_size_decay = conf.TRAIN_SPEC.step_size_decay
    gamma = conf.TRAIN_SPEC.gamma
    episodes = conf.TRAIN_SPEC.episodes
    trajectory_length = conf.TRAIN_SPEC.trajectory_length
    num_envs = conf.TRAIN_SPEC.num_envs
    device = conf.TRAIN_SPEC.device
    tb_logging_name = conf.LOGGING.tb_name

    image_shape = conf.INPUT_SPEC.training_image_shape
    num_frames = conf.INPUT_SPEC.num_frames
    image_shape[2] = image_shape[2] * num_frames
    herke = conf.TRAIN_SPEC.herke

    # get environment specs
    env = conf.ENV_INFO.name
    action_space = gym.make(env).action_space

    # get ppo hyperparams
    ppo_epochs = conf.TRAIN_SPEC.ppo_epochs
    batch_size = conf.TRAIN_SPEC.batch_size
    clip_param = conf.TRAIN_SPEC.clip_param
    value_loss_coeff = conf.TRAIN_SPEC.value_loss_coeff
    entropy_coeff = conf.TRAIN_SPEC.entropy_coeff
    image_loss_coeff = conf.TRAIN_SPEC.image_loss_coeff
    max_grad_norm = conf.TRAIN_SPEC.max_grad_norm

    # initialize handlers
    handlers = []

    if 'HANDLERS' in conf:
        raise NotImplementedError('No handlers are implemented yet')

    # initialize model and trainer
    model = ObjectDetectionNetwork(
        conf.INPUT_SPEC.training_image_shape, 
        action_space, 
        conf.MONET,
        herke=herke,
        load_pretrained=conf.MODEL_SPEC.load_pretrained_object_model)
    agent = ACAgent(model, action_space, gamma, trajectory_length,
                    num_envs=num_envs, device=device)
    trainer = RLPPOTrainer(agent, env, step_size, optimizer, episodes,
                          trajectory_length, gamma, step_size_decay,
                          num_frames, ppo_epochs, batch_size, 
                          clip_param, value_loss_coeff, entropy_coeff,
                          image_loss_coeff, max_grad_norm, 
                          num_envs=num_envs, monet_tb_name=tb_logging_name)

    # register all handlers
    handlers = [TensorboardHandler(logdir='logs/' + tb_logging_name, reset_logdir=True)]
    for h in handlers:
        trainer.register_handler(h)

    return trainer.train(conf)


def build_a2c_experiment(conf):
    """
    Builds an rl experiment runner from the provided config attribute
    :param conf: a parsed config
    :return: a ready trainer
    """
    
    # read all values from config
    embedding_size = conf.INPUT_SPEC.embedding_size
    optimizer = get_optimizer(conf.TRAIN_SPEC.optimizer)
    step_size = conf.TRAIN_SPEC.step_size
    step_size_decay = conf.TRAIN_SPEC.step_size_decay
    gamma = conf.TRAIN_SPEC.gamma
    episodes = conf.TRAIN_SPEC.episodes
    norm_clip = conf.TRAIN_SPEC.norm_clip
    trajectory_length = conf.TRAIN_SPEC.trajectory_length
    num_envs = conf.TRAIN_SPEC.num_envs
    device = conf.TRAIN_SPEC.device
    tb_logging_name = conf.LOGGING.tb_name

    image_shape = conf.INPUT_SPEC.training_image_shape
    num_frames = conf.INPUT_SPEC.num_frames
    image_shape[2] = image_shape[2] * num_frames

    # get environment specs
    env = conf.ENV_INFO.name
    action_space = gym.make(env).action_space

    # initialize handlers
    handlers = []

    if 'HANDLERS' in conf:
        raise NotImplementedError('No handlers are implemented yet')

    # initialize model and trainer
    model = ObjectDetectionNetwork(
        conf.INPUT_SPEC.training_image_shape, 
        action_space, 
        conf.MONET,
        load_pretrained=conf.MODEL_SPEC.load_pretrained_object_model)
    agent = ACAgent(model, action_space, gamma, trajectory_length,
                    num_envs=num_envs, device=device)
    trainer = RLACTrainer(agent, env, step_size, optimizer, episodes,
                          trajectory_length, gamma, step_size_decay,
                          num_frames, norm_clip,
                          num_envs=num_envs, monet_tb_name=tb_logging_name)

    # register all handlers
    handlers = [TensorboardHandler(logdir='logs/' + tb_logging_name, reset_logdir=True)]
    for h in handlers:
        trainer.register_handler(h)

    return trainer.train(conf)


def full_rl_experiment(args):
    parser = ConfigGenerator(args[1])
    conf = parser(argv[2:])
    if conf.EXPERIMENT.algorithm == 'a2c':
        return build_a2c_experiment(conf)
    elif conf.EXPERIMENT.algorithm == 'ppo':
        return build_ppo_experiment(conf)
    else:
        raise ValueError('Experiment not implemented')


if __name__ == "__main__":
    torch.set_num_threads(10)
    full_rl_experiment(argv)
