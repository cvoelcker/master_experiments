"""
Module provides utility functions to access gym objects
"""

import os
import gzip
import dill

import tqdm

import gym
import numpy as np
import torch
from torch import optim
import cv2


def initialize_env(game):
    env = gym.make(game)
    env.reset()
    return env


def load_rl_data(conf):
    return conf.ENV_INFO.name


def create_static_dataset(game, size, file_name,
                          save_path="../data/static_gym/", chunck_size=2 ** 10,
                          agent=None):
    env = initialize_env(game)
    if not agent:
        agent = RandomAgent(env)
    all_obs = []

    os_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    full_path = os_path + save_path + file_name

    print("Creating dataset at location: {}".format(full_path))

    chuncks = int(size / chunck_size) + 1
    print(
        "Dataset will be split in {} chuncks of size {}".format(chuncks, size))

    for x in tqdm.tqdm(range(chuncks)):
        for i in tqdm.tqdm(range(chunck_size)):
            obs, _, done, _ = env.step(agent.act())
            # if np.mean(obs) > 100:
            #     env.reset()
            #     obs, _, done, _ = env.step(agent.act())
            all_obs.append(obs)
            if done:
                env.reset()
            if x * chunck_size + i > size:
                break
        yield np.array(all_obs)
        all_obs = []

    env.close()


def save_numpy_to_file(array, file_name,
                       save_path="../data/static_gym_phoenix/"):
    os_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    full_path = os_path + save_path + file_name
    with gzip.open(full_path, 'wb') as f:
        dill.dump(array, f)


def read_file_to_numpy(full_path):
    with gzip.open(full_path, 'rb') as f:
        arr = dill.load(f)
    return arr


def generate_random_crop(images, crops=(48, 48), crops_per_image=10,
                         remove_all_black=True):
    images_shape = (images.shape[1], images.shape[2])
    new_crops = []
    for image in tqdm.tqdm(images):
        crop = 0
        while crop < crops_per_image:
            x = np.random.randint(0, images_shape[0] - crops[0])
            y = np.random.randint(0, images_shape[1] - crops[1])
            new_crop = image[x:x + crops[0], y:y + crops[1], :]
            # print(np.sum(new_crop))
            # print(not remove_all_black or not np.isclose(np.sum(new_crop), 0))
            if not remove_all_black or not np.isclose(np.sum(new_crop), 0):
                new_crops.append(image[x:x + crops[0], y:y + crops[1], :])
                crop += 1
    return np.array(new_crops)


class GymAgent():
    """
    Interacts with gym environments
    """
    pass


class RandomAgent(GymAgent):
    """
    Only samples random actions from gym environment
    """

    env = None

    def __init__(self, env):
        self.env = env

    def act(self):
        return self.env.action_space.sample()


def get_optimizer(name):
    if name == 'Adam':
        return optim.Adam
    elif name == 'Adadelta':
        return optim.Adadelta
    elif name == 'RMSprop':
        return optim.RMSprop
    else:
        raise NotImplementedError('Unknown optimizer')


class PytorchImage(gym.ObservationWrapper):
    def __init__(self, env, num_frames=1, downsampling=True,
                 sample_shape=(100, 80)):
        super(PytorchImage, self).__init__(env)
        current_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(current_shape[-1] * num_frames, current_shape[0],
                   current_shape[1]),
            dtype=np.float64)
        self.num_frames = num_frames
        self.frames = [np.zeros(current_shape)] * (num_frames - 1)
        self.downsampling = downsampling
        self.sample_shape = sample_shape
        if downsampling:
            self.frames = [np.zeros((
                self.sample_shape[1], self.sample_shape[0],
                current_shape[2]))] * (num_frames - 1)

    def downsample(self, img):
        return cv2.resize(img, self.sample_shape)

    def observation(self, observation):
        return torch.from_numpy(np.swapaxes(observation, 2, 0)).float()

    def state(self):
        next_frame = self.env.env._get_obs()
        if self.downsampling:
            next_frame = self.downsample(next_frame)
        self.frames.append(next_frame)
        self.frames = self.frames[-self.num_frames:]
        state = torch.cat(
            [self.observation(f).unsqueeze(0) for f in self.frames], 1) / 255.0
        return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Supply the parameters')
    parser.add_argument('-env', default='DemonAttack-v0', type=str)
    parser.add_argument('-f', '--file', default='test.pklz', type=str)
    parser.add_argument('-p', '--save_path', default='../data/static_gym/',
                        type=str)
    parser.add_argument('-s', '--size', default=10000, type=int)

    args = parser.parse_args()

    print('Creating new static image set')
    game = args.env
    size = args.size
    file_name = args.file
    save_path = args.save_path

    for i, images in enumerate(
            create_static_dataset(game, size, file_name, save_path=save_path,
                                  chunck_size=2 ** 8)):
        # cropped = generate_random_crop(images)
        save_numpy_to_file(images, file_name + '_{}.gz'.format(i), save_path=save_path)

    print('Conducting sanity check...')

    os_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    full_path = os_path + save_path + file_name + '_0.gz'

    arr = read_file_to_numpy(full_path)
    assert arr is not None
    print('Found file and loaded')
    print('Shape of first array: {}'.format(arr.shape))
