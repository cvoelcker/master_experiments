import torch
from sys import argv

from src.build_run_experiment import full_rl_experiment

if __name__ == "__main__":
    torch.set_num_threads(1)
    full_rl_experiment(argv)
