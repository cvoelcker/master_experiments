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

from trainer import RLTrainer, MONetTrainer
from models.monet_stove import MONetStove

from get_model import get_latent_model, get_rl_models, get_slac
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
policy, qnet, qnet2 = get_rl_models(config, load_run, run_name, run_number)
policy = policy.cuda()
qnet = qnet.cuda()
qnet2 = qnet2.cuda()

slac = get_slac(config, monet, qnet, qnet2, policy)

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA
training_config = config.TRAINING

#############################################
# potential MONet STOVE Pretrain for better initialization
env = AvoidanceTask(BillardsEnv(), action_force=config.RL.action_force)

source_loader = generators.FunctionLoader(
        generate_envs_data,
        {'env': env, 'num_runs': 200, 'run_len': 100})

transformers = [
        transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
        transformers.TypeTransformer(config.EXPERIMENT.device)
        ]

data = SequenceDictDataSet(source_loader, transformers, 8)
 
if config.EXPERIMENT.pretrain_model:
   
    trainer = setup_trainer(MONetTrainer, monet, training_config, data)
    
    checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
    trainer.register_handler(checkpointing)
    
    regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'), log_name_list=['imgs'])
    trainer.register_handler(regular_logging)
    
    tb_logging_list = ['average_elbo', 'trans_lik', 'log_z_f', 'img_lik_forward', 'elbo', 'z_s', 'img_lik_mean', 'p_x_loss', 'p_x_loss_mean']
    tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
    trainer.register_handler(tb_logger)
    
    trainer.model.img_model.init_background_weights(trainer.train_dataloader.dataset.get_all())
    
    trainer.train(config.TRAINING.epochs, train_only=True, pretrain=config.EXPERIMENT.pretrain_img)
    if config.EXPERIMENT.pretrain_img:
            monet.img_model.beta = config.MODULE.MONET.beta
            trainer = setup_trainer(MONetTrainer, monet, training_config, data)

            checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
            trainer.register_handler(checkpointing)

            regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'), log_name_list=['imgs'])
            trainer.register_handler(regular_logging)

            tb_logging_list = ['average_elbo', 'trans_lik', 'log_z_f', 'img_lik_forward', 'elbo', 'z_s', 'img_lik_mean', 'p_x_loss_mean']
            tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
            trainer.register_handler(tb_logger)

            trainer.train(config.TRAINING.epochs * 10, train_only=True, pretrain=False)


##################################################################################
# build RL full trainer
env.reset()

memory = buffer.StateBuffer(500000, config.DATA.shape, 7, config.MODULE.DYNAMICS.action_space)
for i in range(200):
    memory.put(data.dataset['X'][i], 
    np.argmax(data.dataset['action'][i], -1).reshape(-1),
    data.dataset['reward'][i].reshape(-1), 
    data.dataset['done'][i].reshape(-1))

trainer = RLTrainer(config.RL, slac, env, memory)

checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
trainer.register_handler(checkpointing)

regular_logging = file_handler.EpochFileHandler(os.path.join(run_path, 'data'), log_name_list=['imgs'])
trainer.register_handler(regular_logging)

tb_logging_list = ['q1', 'q2', 'p', 'e', 'm', 'ent', 'rolling_reward']
tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=tb_logging_list)
trainer.register_handler(tb_logger)

trainer.train(config.TRAINING.total_rl_steps, config.TRAINING.rl_batch_size, config.MODULE.SLAC.debug)