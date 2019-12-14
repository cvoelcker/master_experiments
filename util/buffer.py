from typing import Tuple

import torch
from torch.utils.data import Dataset

import numpy as np
import scipy
from PIL import Image


class StateBuffer(Dataset):

    def __init__(self, buffer_size: int, image_size: Tuple[int, int, int], seq_len: int, action_space: int):
        self.s_buffer = np.zeros((buffer_size, *image_size), dtype=np.uint8)
        self.a_buffer = np.zeros((buffer_size,), dtype=np.uint8)
        self.r_buffer = np.zeros((buffer_size,), dtype=np.int8)
        self.d_buffer = np.zeros((buffer_size,), dtype=np.uint8)
        self.fill = 0
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.action_space = action_space

        self.total_samples = 0

    def __len__(self):
        return self.fill - self.seq_len

    def __getitem__(self, idx: int):
        assert idx < (self.fill - self.seq_len) and (self.fill >= self.seq_len)
        x = self.torch_transform(self.s_buffer[idx:idx+self.seq_len]).float() / 255.0
        a = self.torch_transform(self.a_buffer[idx:idx+self.seq_len], action_expand=True)
        a_idx = self.torch_transform(self.a_buffer[idx:idx+self.seq_len]).long()
        r = self.torch_transform(self.r_buffer[idx:idx+self.seq_len]).float()
        d = self.torch_transform(self.d_buffer[idx:idx+self.seq_len]).float()
        return {'x': x, 'a': a, 'r': r, 'd': d, 'a_idx': a_idx}

    def put(self, obs: np.array, actions: np.array, rewards: np.array, dones: np.array):
        new_len = len(obs)
        offset = self.buffer_size - new_len - self.fill
        if offset < 0:
            self.s_buffer[0:self.fill+offset] = self.s_buffer[-offset:self.fill]
            self.a_buffer[0:self.fill+offset] = self.a_buffer[-offset:self.fill]
            self.r_buffer[0:self.fill+offset] = self.r_buffer[-offset:self.fill]
            self.d_buffer[0:self.fill+offset] = self.d_buffer[-offset:self.fill]
            self.fill += offset
        reshaped_img = np.array([self.resize(o) for o in obs])
        self.s_buffer[self.fill:self.fill+new_len] = reshaped_img
        self.a_buffer[self.fill:self.fill+new_len] = actions
        self.r_buffer[self.fill:self.fill+new_len] = rewards
        self.d_buffer[self.fill:self.fill+new_len] = dones
        self.fill += new_len
        self.total_samples += new_len

    def clean(self):
        self.fill = 0

    def resize(self, obs):
        return np.array(Image.fromarray(obs.astype(np.uint8)).resize(self.image_size[:2]))

    def torch_transform(self, x, action_expand=False):
        if len(x.shape) >= 4:
            return torch.Tensor(np.moveaxis(x, -1, -3)).float().cuda()
        elif action_expand:
            # first, convert to torch tensor, add an additional dimension (the one_hot dimension)
            # and convert to long, since torch expects long as indices
            idx = torch.Tensor(x).unsqueeze(-1).long()
            # create the output vector and expand one_hot dimension
            if len(idx.shape) == 2:
                one_hot = torch.zeros_like(idx).repeat(1, self.action_space)
            if len(idx.shape) == 3:
                one_hot = torch.zeros_like(idx).repeat(1, 1, self.action_space)
            # adress one_hot vector with the action vector
            one_hot = one_hot.scatter(-1, idx, 1.)
            return one_hot.float().cuda()
        else:
            return torch.Tensor(x).float().cuda()