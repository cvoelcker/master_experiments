import os
import pickle

import gym
import torch
import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm

from config_parser.config_parser import ConfigGenerator

from torch_runner.experiment_setup import setup_experiment, load_config, get_run_path
from torch_runner.data import transformers, file_loaders, generators
from torch_runner.data.base import BasicDataSet, SequenceDataSet, SequenceDictDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from trainer import MONetTrainer, MONetTester
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

config, config_object = setup_experiment(config, config_object, debug=False)
monet = get_model(config, MONetStove).cuda()

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

monet = monet.img_model

dir_path = 'experiments/billards/run_042/data/'

for i, f in tqdm(enumerate(os.listdir(dir_path))):
    if f[0] == 'z':
        d = pickle.load(open(os.path.join(dir_path, f), 'rb'))
        for x in range(7):
            with torch.no_grad():
                rec = monet.reconstruct_from_latent(d[:, x].cuda().float())
            torch.save(rec.detach().cpu().numpy(), f'vis_{i}_{x}')
