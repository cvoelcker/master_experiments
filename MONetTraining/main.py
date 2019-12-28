import os

from torchvision import transforms
import torch
import gym

from config_parser.config_parser import ConfigGenerator
from spatial_monet.spatial_monet import MaskedAIR

from torch_runner.experiment_setup import setup_experiment, load_config, get_model, get_run_path
from torch_runner.data import transformers, file_loaders, generators
from torch_runner.data.base import BasicDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from trainer import MONetTrainer

from util.data import generate_envs_data
from util.envs import AvoidanceTask, BillardsEnv

config, config_object = load_config()

monet = get_model(config, MaskedAIR).cuda()
config, config_object = setup_experiment(config, config_object, debug=False)

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
        lambda **kwargs: generate_envs_data(**kwargs)['X'].squeeze(),
        {'env': env, 'num_runs': 1, 'run_len': 25000})

data_transformers = [
        transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
        transformers.TypeTransformer(config.EXPERIMENT.device)
        ]
print('Loading data')
data = BasicDataSet(source_loader, data_transformers)

trainer = setup_trainer(MONetTrainer, monet, training_config, data)
checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
trainer.register_handler(checkpointing)
tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=['loss', 'kl_loss', 'mask_loss', 'p_x_loss', 'mse', 'p_x_loss_mean'])
trainer.register_handler(tb_logger)
regular_logging = file_handler.NStepFileHandler(150, os.path.join(run_path, 'data'), log_name_list=['reconstruction', 'masks'])
trainer.register_handler(regular_logging)

# MONet init block
# for w in trainer.model.parameters():
#     std_init = 0.01
#     torch.nn.init.normal_(w, mean=0., std=std_init)
trainer.model.init_background_weights(trainer.train_dataloader.dataset.get_all())
trainer.check_ready()
trainer.train(config.TRAINING.epochs, train_only=True, visdom=True)
