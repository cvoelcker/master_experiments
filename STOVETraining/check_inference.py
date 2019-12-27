import os

import gym
import torch
import numpy as np
from torchvision import transforms
import torch

from config_parser.config_parser import ConfigGenerator

from torch_runner.experiment_setup import setup_experiment, load_config, get_run_path
from torch_runner.data import transformers, file_loaders, generators
from torch_runner.data.base import BasicDataSet, SequenceDataSet, SequenceDictDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from trainer import MONetTrainer, MONetTester, MONetInferenceTester
from models.monet_stove import MONetStove

from get_model import get_model
from util.data import generate_envs_data
from util.envs import AvoidanceTask, BillardsEnv

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
monet = get_model(config, MONetStove, load_run, run_name, run_number).cuda()

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA
training_config = config.TRAINING

if config.EXPERIMENT.game == 'billards':
    env = AvoidanceTask(BillardsEnv(), action_force=0.6)
else:
    env = gym.make(config.EXPERIMENT.game)

source_loader = generators.FunctionLoader(
        generate_envs_data,
        {'env': env, 'num_runs': 1, 'run_len': 500})

transformers = [
        transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
        transformers.TypeTransformer(config.EXPERIMENT.device)
        ]

data = SequenceDictDataSet(source_loader, transformers, 50)

trainer = setup_trainer(MONetInferenceTester, monet, training_config, data)

checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
trainer.register_handler(checkpointing)
regular_logging = file_handler.StepFileHandler(os.path.join(run_path, 'data'), log_name_list=['z', 'r', 'imgs', 'imgs_inferred'])
trainer.register_handler(regular_logging)
trainer.register_handler(tb_logger)

trainer.train(config.TRAINING.epochs, train_only=True)

