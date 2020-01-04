import os

from torchvision import transforms
import torch
from torch import nn
import gym

from config_parser.config_parser import ConfigGenerator
from spatial_monet.spatial_monet import MaskedAIR
from spatial_monet.genesis import GENESIS
from spatial_monet.monet import Monet
from models.baseline_broadcast_vae import BroadcastVAE

from torch_runner.experiment_setup import setup_experiment, load_config, get_model, get_run_path
from torch_runner.data import transformers, file_loaders, generators
from torch_runner.data.base import BasicDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from trainer import MONetTrainer
from util.data import generate_envs_data

all_games = [
        'assault',
        'demon_attack',
        'boxing',
        'skiing',
        'double_dunk',
        'kung_fu_master',
        'phoenix',
        'beam_rider',
        'asteroids',
        'bank_heist'
        ]

all_games = [''.join((s.capitalize() for s in g.split('_'))) + '-v0' for g in all_games]

config, config_object = load_config()

config, config_object = setup_experiment(config, config_object, debug=False)

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA
training_config = config.TRAINING

print('running second half')
l = len(all_games)
for game in all_games:
        print('Running {}'.format(game))
        monet = nn.DataParallel(BroadcastVAE(**config.MODULE._asdict())).cuda()
        # monet = nn.DataParallel(Monet(config.MODULE, 128, 128)).cuda()
        print('Generated model')
        env = gym.make(game)
        
        source_loader = generators.FunctionLoader(
                lambda **kwargs: generate_envs_data(**kwargs)['X'].squeeze(),
                {'env': env, 'num_runs': 1, 'run_len': 25000})

        data_transformers = [
                transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
                transformers.TypeTransformer(config.EXPERIMENT.device)
                ]
        print('Loading data')
        data = BasicDataSet(source_loader, data_transformers)
        print('Setting up trainer')
        trainer = setup_trainer(MONetTrainer, monet, training_config, data)
        check_path = os.path.join(run_path, 'checkpoints_{}'.format(game))
        if not os.path.exists(check_path):
                os.mkdir(check_path)
        checkpointing = file_handler.EpochCheckpointHandler(check_path)
        trainer.register_handler(checkpointing)
        log_path = os.path.join(run_path, 'logging_{}'.format(game))
        if not os.path.exists(log_path):
                os.mkdir(log_path)
        tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging_{}'.format(game), log_name_list=['loss', 'kl_loss', 'mask_loss', 'p_x_loss'])
        print('Running training')
        trainer.register_handler(tb_logger)
        
        # MONet init block
        # for w in trainer.model.parameters():
        #     std_init = 0.01
        #     torch.nn.init.normal_(w, mean=0., std=std_init)
        # trainer.model.init_background_weights(trainer.train_dataloader.dataset.get_all())
        
        trainer.train(config.TRAINING.epochs, train_only=True, visdom=False)
        env.close()
