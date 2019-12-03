import os

import gym

from .monitor import Monitor
from .multiprocessing_env import SubprocVecEnv, VecPyTorch, \
    VecPyTorchFrameStack
from .wrappers import *


def make_env(env_name, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_name)
        env.seed(seed + rank)
        assert "NoFrameskip" in env_name, \
            "Require environment with no frameskip"
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

        allow_early_resets = True
        assert log_dir is not None, \
            "Log directory required for Monitor! (which is required " \
            "for episodic return reporting)"
        try:
            os.mkdir(log_dir)
        except:
            pass
        env = Monitor(env, os.path.join(log_dir, str(rank)),
                      allow_early_resets=allow_early_resets)

        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk


def make_envs(env_name, num_envs, seed, device, log_dir):
    envs = [make_env(env_name, seed, i, log_dir) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, device)
    envs = VecPyTorchFrameStack(envs, 4, device)
    return envs
