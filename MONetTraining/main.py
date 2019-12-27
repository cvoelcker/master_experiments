import os

from torchvision import transforms
import torch

from config_parser.config_parser import ConfigGenerator
from spatial_monet.spatial_monet import MaskedAIR

from torch_runner.experiment_setup import setup_experiment, load_config, get_model, get_run_path
from torch_runner.data import transformers, file_loaders
from torch_runner.data.base import BasicDataSet
from torch_runner.training_setup import setup_trainer
from torch_runner.handlers import file_handler, tb_handler

from trainer import MONetTrainer

config, config_object = load_config()

monet = get_model(config, MaskedAIR).cuda()
config, config_object = setup_experiment(config, config_object, debug=False)

run_path = get_run_path(
        config.EXPERIMENT.experiment_dir,
        config.EXPERIMENT.run_name,
        config.EXPERIMENT.run_number)

data_config = config.DATA
training_config = config.TRAINING

source_loader = file_loaders.DirectoryLoader(
        directory=data_config.data_dir, 
        compression_type='gzip')

transformers = [
        transformers.TorchVisionTransformerComposition(config.DATA.transform, config.DATA.shape),
        transformers.TypeTransformer(config.EXPERIMENT.device)
        ]

data = BasicDataSet(source_loader, transformers)

trainer = setup_trainer(MONetTrainer, monet, training_config, data)
checkpointing = file_handler.EpochCheckpointHandler(os.path.join(run_path, 'checkpoints'))
trainer.register_handler(checkpointing)
tb_logger = tb_handler.NStepTbHandler(config.EXPERIMENT.log_every, run_path, 'logging', log_name_list=['loss', 'kl_loss', 'mask_loss', 'p_x_loss'])
trainer.register_handler(tb_logger)

trainer.model.init_background_weights(trainer.train_dataloader.dataset.get_all())

trainer.train(config.TRAINING.epochs, train_only=True)
