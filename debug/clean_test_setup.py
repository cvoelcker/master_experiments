import sys

from config_parser.config_parser import ConfigGenerator
from torch_runner.experiment_setup import setup_experiment, load_config, clean_experiment_directory

config, config_object = load_config()
print(config)
clean_experiment_directory(config)
