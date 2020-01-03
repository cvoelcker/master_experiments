import pickle
import torch
import numpy as np

counter = 0
i = 20
while counter < 200:
    try:
        x = []
        for j in range(1,41):
            x.append(pickle.load(open('experiments/slac-model/run_{:03d}/data/mse_{:07d}.save'.format(i, j), 'rb')))
        print(torch.cat(x).mean().item())
        counter += 1
    except Exception as e:
        print(e)
        pass
    i += 1
