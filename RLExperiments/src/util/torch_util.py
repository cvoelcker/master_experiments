from torch import optim
import torch.nn as nn


def get_optimizer(name):
    if name == 'Adam':
        return optim.Adam
    elif name == 'Adadelta':
        return optim.Adadelta
    elif name == 'RMSprop':
        return lambda p, lr: optim.RMSprop(p, lr, 0.99, 1e-5)
    else:
        raise NotImplementedError('Unknown optimizer')


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
