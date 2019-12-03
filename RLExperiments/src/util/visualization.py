import os
import matplotlib
import dill
import gzip

import numpy as np
import torchvision.transforms as transforms

from PIL import Image


def load_gameframe(file_name, save_dir='../data/static_gym/'):
    imgpath = os.path.join(save_dir, file_name)
    with gzip.open(imgpath, 'rb') as f:
        img = dill.load(f)
        return img


def visualize_gameframe(frame, shape=None, file_name='gym_picture.png', save_dir='../data/images_gym/'):
    img = Image.fromarray(frame)
    if shape:
        img = subsample_gameframe(img, shape)
    imgpath = os.path.join(save_dir, file_name)
    img.save(imgpath)


def subsample_gameframe(frame, shape):
    transform = transforms.Compose([transforms.Resize(shape)])
    return transform(frame)


if __name__ == "__main__":
    frames = load_gameframe('test.pklz_0.gz')
    print(frames.shape)
    for i in range(1000):
        visualize_gameframe(frames[i], file_name="original_{}.png".format(i))
        # visualize_gameframe(frames[11], shape=(105,80))
