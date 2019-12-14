import os

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

from trainer import MONetTrainer
from models.monet_stove import MONetStove

from get_model import get_model
from util.data import generate_envs_data

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config, config_object = load_config()
print(config)

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
training_config = config.TRAINING

if config.EXPERIMENT.experiment == 'billards':
        source_loader = file_loaders.FileLoader(
                file_name=data_config.data_dir, 
                compression_type='pickle')
        # env = gym.make('DemonAttack-v0')
        # source_loader = generators.FunctionLoader(
        #         generate_envs_data,
        #         {'env': env, 'num_runs': 100, 'run_len': 100})

        transformers = [
                transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
                transformers.TypeTransformer(config.EXPERIMENT.device)
                ]

elif config.EXPERIMENT.experiment == 'atari':
        env = gym.make('DemonAttack-v0')
        source_loader = generators.FunctionLoader(
                generate_envs_data,
                {'env': env, 'num_runs': 200, 'run_len': 100})

        transformers = [
                transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
                transformers.TypeTransformer(config.EXPERIMENT.device)
                ]

data = SequenceDictDataSet(source_loader, transformers, 8)

trainer = setup_trainer(MONetTrainer, monet, training_config, data)

checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
trainer.register_handler(checkpointing)

regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'), log_name_list=['imgs'])
trainer.register_handler(regular_logging)

tb_logging_list = ['average_elbo', 'trans_lik', 'log_z_f', 'img_lik_forward', 'elbo', 'z_s', 'img_lik_mean', 'p_x_loss', 'p_x_loss_mean']
tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
trainer.register_handler(tb_logger)

trainer.model.img_model.init_background_weights(trainer.train_dataloader.dataset.get_all())

trainer.train(config.TRAINING.epochs, train_only=True, pretrain=config.TRAINING.pretrain)

if config.TRAINING.pretrain:
        monet.img_model.beta = config.MODULE.MONET.beta
        trainer = setup_trainer(MONetTrainer, monet, training_config, data)
        
        checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
        trainer.register_handler(checkpointing)
        
        regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'), log_name_list=['imgs'])
        trainer.register_handler(regular_logging)
        
        tb_logging_list = ['average_elbo', 'trans_lik', 'log_z_f', 'img_lik_forward', 'elbo', 'z_s', 'img_lik_mean', 'p_x_loss_mean']
        tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
        trainer.register_handler(tb_logger)

        trainer.train(config.TRAINING.epochs * 20, train_only=True, pretrain=False)
