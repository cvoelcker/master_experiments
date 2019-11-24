import sys

from config_parser.config_parser import ConfigGenerator
from torch_runner.experiment_setup import setup_experiment, load_config

config, config_object = load_config()
print(config)
config, config_object = setup_experiment(config, config_object, debug=False)
print(config)
