from sys import argv
import os
import pickle
from tqdm import tqdm

from torchvision import transforms
import torch
from torch import nn
import numpy as np
import gym

from config_parser.config_parser import ConfigGenerator
from spatial_monet.spatial_monet import MaskedAIR
from spatial_monet.monet import Monet
from models.baseline_broadcast_vae import BroadcastVAE

from torch_runner.experiment_setup import setup_experiment, load_config, get_model, get_run_path
from torch_runner.data import transformers, file_loaders, generators
from torch_runner.data.base import BasicDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from util.data import generate_envs_data


class MONetEvaluator():

    def __init__(self, data, models):
        self.data = data
        self.models = models
    
    def numpify(self, img):
        img = img.cpu().detach().numpy()
        img = np.moveaxis(img, -3, -1)
        img = img * 255.
        img = img.astype(np.uint8)
        return img

    def evaluate(self):
        with torch.no_grad():
            all_imgs = {}
            all_recons = {}
            all_losses = {}
            for key in self.models.keys():
                print(key)
                imgs = []
                recons = []
                losses = []
                for d in tqdm(self.data):
                    img = self.numpify(d)
                    imgs.append(img)
                    model = self.models[key]
                    _, results = model(d.unsqueeze(0))
                    recon = self.numpify(results['reconstruction'])
                    recons.append(recon)
                    loss = np.mean((img - recon) ** 2)
                    losses.append(loss)
                all_imgs[key] = imgs
                all_recons[key] = recons
                all_losses[key] = losses
        return all_losses, all_recons, all_imgs


def main():
    all_games = [
            # 'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber', 
            # 'defender',  ## apparently, this is really broken
            # 'demon_attack', 'double_dunk', 'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 
            # 'phoenix', 
            'demon_attack',
            # 'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'
            ]
            
    
    all_games = [''.join((s.capitalize() for s in g.split('_'))) + '-v0' for g in all_games]
    
    monet_config = ConfigGenerator('trained_baselines/monet/config.yml')
    spatial_monet_config = ConfigGenerator('experiments/demon-attack/run_000/config.yml')
    vae_config = ConfigGenerator('trained_baselines/vae/config.yml')
    monet_config = monet_config(argv[1:])
    spatial_monet_config = spatial_monet_config(argv[1:])
    # genesis_config =
    vae_config = vae_config(argv[1:])
    for game in all_games:
        print()
        print()
        print(f'Running {game}')
    
        # monet_checkpoint = f'trained_baselines/monet/checkpoints_{game}/model_state_0000050.save'
        spatial_monet_checkpoint = f'experiments/demon-attack/run_000/checkpoints_{game}/model_state_0000020.save'
        # genesis_checkpoint = f'trained_baselines/monet/checkpoints_{game}/model_state_0000050.save'
        # vae_checkpoint = f'trained_baselines/vae/checkpoints_{game}/model_state_0000050.save'
    
        # monet = nn.DataParallel(Monet(**monet_config.MODULE._asdict())).cuda()
        # monet.load_state_dict(torch.load(monet_checkpoint))
    
        spatial_monet = MaskedAIR(**spatial_monet_config.MODULE._asdict()).cuda()
        # spatial_monet = nn.DataParallel(MaskedAIR(**spatial_monet_config.MODULE._asdict())).cuda()
        spatial_monet.load_state_dict(torch.load(spatial_monet_checkpoint))
    
        # genesis = GENESIS(**genesis_config._asdict()).cuda
        # genesis.load_state_dict(torch.load(genesis_checkpoint))
    
        # vae = nn.DataParallel(BroadcastVAE(**vae_config.MODULE._asdict())).cuda()
        # vae.load_state_dict(torch.load(vae_checkpoint))
    
        all_models = {
            # 'monet': monet,
            'spatial_monet': spatial_monet,
            # 'genesis': genesis,
            # 'vae': vae
        }
    
        env = gym.make(game)
        
        source_loader = generators.FunctionLoader(
                lambda **kwargs: generate_envs_data(**kwargs)['X'].squeeze(),
                {'env': env, 'num_runs': 1, 'run_len': 100})
    
        data_transformers = [
                transformers.TorchVisionTransformerComposition(monet_config.DATA.transform, monet_config.DATA.shape),
                transformers.TypeTransformer(monet_config.EXPERIMENT.device)
                ]
        data = BasicDataSet(source_loader, data_transformers)
    
        evaluator = MONetEvaluator(data, all_models)
        losses, recons, imgs = evaluator.evaluate()
        if not os.path.exists(f'eval/{game}'):
            os.mkdir(f'eval/{game}')
        pickle.dump(losses, open(f'eval/{game}/mse.pkl', 'wb'))
        pickle.dump(recons, open(f'eval/{game}/recons.pkl', 'wb'))
        pickle.dump(imgs, open(f'eval/{game}/imgs.pkl', 'wb'))


if __name__ == '__main__':
    main()
