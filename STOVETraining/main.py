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
from monet_stove import MONetStove

from get_model import get_model
from data import generate_envs_data

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config, config_object = load_config()

if config.EXPERIMENT.random_seed:
    seed = np.random.randint(2**32 - 1)
    config_object.config_dict['EXPERIMENT']['seed'] = seed
    config_object.config_dict['EXPERIMENT']['random_seed'] = False
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    np.random.seed(config.EXPERIMENT.seed)
    torch.manual_seed(config.EXPERIMENT.seed)

monet = get_model(config, MONetStove).cuda()
config, config_object = setup_experiment(config, config_object, debug=False)

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA
training_config = config.TRAINING

# source_loader = file_loaders.DirectoryLoader(
#         directory=data_config.data_dir, 
#         compression_type='pickle', 
#         preprocess_function = lambda x: x['X'])
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
regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'))
trainer.register_handler(regular_logging)
tb_logging_list = ['average_elbo', 'trans_lik', 'log_z_f', 'img_lik_forward', 'img_lik_mean', 'elbo', 'z_s']
tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
trainer.register_handler(tb_logger)

trainer.train(config.TRAINING.epochs, train_only=True)

