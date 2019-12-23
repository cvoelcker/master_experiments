import os
import pickle

import gym
import torch
import numpy as np
from torchvision import transforms

from config_parser.config_parser import ConfigGenerator

from torch_runner.experiment_setup import setup_experiment, load_config, get_run_path
from torch_runner.data import transformers, file_loaders, generators
from torch_runner.data.base import BasicDataSet, SequenceDataSet, SequenceDictDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from trainer import SLACTrainer, SACTrainer, MONetTrainer
from models.monet_stove import MONetStove

from get_model import get_latent_model, get_rl_models, get_slac, get_sac
from util.data import generate_envs_data
from util import buffer
from util.envs import BillardsEnv, AvoidanceTask

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config, config_object = load_config()

load_run = config.EXPERIMENT.load_run
run_name = config.EXPERIMENT.run_name
run_number = config.EXPERIMENT.run_number

if config.EXPERIMENT.random_seed:
    seed = np.random.randint(2**32 - 1)
    config_object.config_dict['EXPERIMENT']['seed'] = seed
    config_object.config_dict['EXPERIMENT']['random_seed'] = False
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    np.random.seed(config.EXPERIMENT.seed)
    torch.manual_seed(config.EXPERIMENT.seed)

config, config_object = setup_experiment(config, config_object, debug=False)

monet = get_latent_model(config, MONetStove, load_run, run_name, run_number).cuda()

if config.EXPERIMENT.algorithm == 'slac':
    policy, qnet, qnet2 = get_rl_models(config, load_run, run_name, run_number)
    slac = get_slac(config, monet, qnet, qnet2, policy)
elif config.EXPERIMENT.algorithm == 'sac':
    policy, qnet, qnet2 = get_rl_models(config, load_run, run_name, run_number)
    slac = get_sac(config, monet, qnet, qnet2, policy)

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA
training_config = config.TRAINING

#############################################
# potential MONet STOVE Pretrain for better initialization
if config.EXPERIMENT.game == 'billards':
    env = AvoidanceTask(BillardsEnv(), action_force=config.RL.action_force)
else:
    env = gym.make(config.EXPERIMENT.game)

source_loader = generators.FunctionLoader(
        generate_envs_data,
        {'env': env, 'num_runs': 200, 'run_len': 100})

transformers = [
        transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
        transformers.TypeTransformer(config.EXPERIMENT.device)
        ]

data = SequenceDictDataSet(source_loader, transformers, 8)

memory = buffer.StateBuffer(500000, config.DATA.shape, 8, config.MODULE.DYNAMICS.action_space)
for i in range(200):
    memory.put(data.dataset['X'][i], 
    np.argmax(data.dataset['action'][i], -1).reshape(-1),
    data.dataset['reward'][i].reshape(-1), 
    data.dataset['done'][i].reshape(-1))

# build RL full trainer
env.reset()

if config.EXPERIMENT.algorithm == 'slac':
    trainer = SLACTrainer(config.RL, slac, env, memory)
elif config.EXPERIMENT.algorithm == 'sac':
    trainer = SACTrainer(config.RL, slac, env, memory)

if config.EXPERIMENT.pretrain_model:
    if config.EXPERIMENT.algorithm == 'sac':
        raise ValueError('Cannot pretrain model-free baseline')
    trainer.pretrain(config.TRAINING.batch_size, config.TRAINING.epochs, img=config.EXPERIMENT.pretrain_img)
    if config.EXPERIMENT.pretrain_img:
        print('Pretraining model')
        trainer.pretrain(config.TRAINING.batch_size, config.TRAINING.epochs * 5, img=False)

checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
trainer.register_handler(checkpointing)

regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'), log_name_list=['imgs'])
trainer.register_handler(regular_logging)

tb_logging_list = ['q1', 'q2', 'p', 'e', 'm', 'ent', 'rolling_reward']
tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
trainer.register_handler(tb_logger)

e_means, e_vars = trainer.train(config.TRAINING.total_steps, config.TRAINING.rl_batch_size, config.MODULE.SLAC.debug)

pickle.dump(e_means, open(os.path.join(run_path, 'means.pkl'), 'wb')
pickle.dump(e_vars, open(os.path.join(run_path, 'vars.pkl'), 'wb')