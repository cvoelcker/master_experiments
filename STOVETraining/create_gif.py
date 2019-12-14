import pickle
import sys
import torch
import numpy as np
import imageio

def make_gif(file_loc):
    with open(file_loc, 'rb') as f:
        run = pickle.load(f)

        for i, seq in enumerate(run):
            print(i)
            all_imgs = seq.numpy().astype(np.uint8)
            all_imgs = np.moveaxis(all_imgs, 1, -1)
            imageio.mimsave(f'./gifs/{i}.gif', all_imgs, fps=24)

file_loc = sys.argv[1]
print(f"Making gif from run at {file_loc}")

make_gif(file_loc)

