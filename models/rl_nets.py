import util.buffer as buffer
import copy

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import pickle


def flatten_probs(action_probs, min_prob=0.001):
    action_space = action_probs.shape[-1]
    action_probs = (min_prob / action_space) + (1 - min_prob) * action_probs
    return action_probs


class GraphHead(nn.Module):
    """
    Taken from VIN style dynamics core
    """
    def __init__(self, num_slots=8, cl=64):
        super().__init__()
        self.num_obj = num_slots
        self.cl = cl

        # Interaction Net Core Modules
        self.state_enc = nn.Linear(cl, cl)
        # Self-dynamics MLP
        self.self_cores = nn.ModuleList()
        self.self_cores.append(nn.Linear(cl, cl))
        self.self_cores.append(nn.Linear(cl, cl))

        # Relation MLP
        self.rel_cores = nn.ModuleList()
        self.rel_cores.append(nn.Linear(1 + cl * 2, 2 * cl))
        self.rel_cores.append(nn.Linear(2 * cl, cl))
        self.rel_cores.append(nn.Linear(cl, cl))

        # Attention MLP
        self.att_net = nn.ModuleList()
        self.att_net.append(nn.Linear(1 + cl * 2, 2 * cl))
        self.att_net.append(nn.Linear(2 * cl, cl))
        self.att_net.append(nn.Linear(cl, 1))

        # Attention mask
        diag_mask = 1 - torch.eye(
            self.num_obj,
            ).unsqueeze(2).unsqueeze(0)
        self.register_buffer('diag_mask', diag_mask)

        self.nonlinear = F.elu

    def forward(self, s):
        # add back positions for distance encoding
        s = torch.cat([s[..., :2], self.state_enc(s)[..., 2:]], -1)

        self_sd_h1 = self.nonlinear(self.self_cores[0](s))
        self_dynamic = self.self_cores[1](self_sd_h1) + self_sd_h1

        object_arg1 = s.unsqueeze(2).repeat(1, 1, self.num_obj, 1)
        object_arg2 = s.unsqueeze(1).repeat(1, s.shape[1], 1, 1)
        distances = (object_arg1[..., 0] - object_arg2[..., 0])**2 +\
                    (object_arg1[..., 1] - object_arg2[..., 1])**2
        distances = distances.unsqueeze(-1)

        # shape (n, o, o, 2cl+1)
        combinations = torch.cat((object_arg1, object_arg2, distances), 3)
        rel_sd_h1 = self.nonlinear(self.rel_cores[0](combinations))
        rel_sd_h2 = self.nonlinear(self.rel_cores[1](rel_sd_h1))
        rel_factors = self.rel_cores[2](rel_sd_h2) + rel_sd_h2

        attention = self.nonlinear(self.att_net[0](combinations))
        attention = self.nonlinear(self.att_net[1](attention))
        # change this to sigmoid for saving the size
        attention = torch.sigmoid(self.att_net[2](attention))

        # mask out object interacting with itself (n, o, o, cl)
        rel_factors = rel_factors * self.diag_mask * attention

        # relational dynamics per object, (n, o, cl)
        rel_dynamic = torch.sum(rel_factors, 2)

        return torch.sum(self_dynamic + rel_dynamic, 1)


class SimpleGraphHead(nn.Module):
    """
    Taken from VIN style dynamics core
    """
    def __init__(self, num_slots=8, cl=64):
        super().__init__()
        self.num_obj = num_slots
        self.cl = cl

        # Interaction Net Core Modules
        self.state_enc = nn.Sequential(
            nn.Linear(cl, cl),
            nn.ReLU(),
            nn.Linear(cl, cl),
            nn.ReLU())
        # Self-dynamics MLP
        self.self_dynamics = nn.Sequential(
            nn.Linear(cl, 2 * cl),
            nn.ReLU(),
            nn.Linear(2 * cl, 2 * cl))
        # Relation MLP
        self.rel_core = nn.Sequential(
            nn.Linear(2 * cl + 1, 2 * cl),
            nn.ReLU(),
            nn.Linear(2 * cl, 2 * cl),
            nn.ReLU(),
            nn.Linear(2 * cl, 2 * cl))
        # Attention MLP
        self.att_net = nn.Sequential(
            nn.Linear(2 * cl + 1, cl),
            nn.ReLU(),
            nn.Linear(cl, 2 * cl))

    def forward(self, s):
        # add back positions for distance encoding
        s = self.state_enc(s)

        # build object_graph
        object_arg1 = s.unsqueeze(2).repeat(1, 1, self.num_obj, 1)
        object_arg2 = s.unsqueeze(1).repeat(1, s.shape[1], 1, 1)
        distances = (object_arg1[..., 0] - object_arg2[..., 0])**2 +\
                    (object_arg1[..., 1] - object_arg2[..., 1])**2
        distances = distances.unsqueeze(-1)
        distances = torch.cat([object_arg1, object_arg2, distances], -1)

        rel_s = self.rel_core(distances)

        # change this to tanh for saving the size
        attention = torch.sigmoid(self.att_net(distances))
        rel_factors = rel_s * attention

        # relational dynamics per object, (n, o, cl)
        rel_dynamic = torch.mean(rel_factors, 2)

        # self evolution
        self_dynamics = self.self_dynamics(s)

        return torch.mean(self_dynamics + rel_dynamic, 1)


class AbstractQNet(nn.Module):
    def __init__(self):
        super().__init__()

    def sample_action(self, s):
        q = F.softmax(self(s), -1)
        # v_function = torch.logsumexp(q, 1, keepdim=True)
        # policy = torch.exp(q - v_function)
        dist = torch.distributions.Categorical(probs = q)
        return dist.sample(), q, torch.log(q)

    def sample_max_action(self, s):
        _, policy, log_policy = self.sample_action(s)
        return torch.argmax(policy, -1), policy, log_policy


class GraphQNet(AbstractQNet):
    def __init__(self, graph_head, action_space, needs_actions = False):
        super().__init__()
        self.graph_head = graph_head
        self.cl = graph_head.cl
        self.action_space = action_space
        self.input_length = 2 * self.cl + action_space if needs_actions else 2 * self.cl
        self.needs_actions = needs_actions
        self.q_head = nn.Sequential(
            nn.Linear(self.input_length, self.cl),
            nn.ReLU(),
            nn.Linear(self.cl, self.cl//2),
            nn.ReLU(),
            nn.Linear(self.cl//2, self.cl//4),
            nn.ReLU(),
            nn.Linear(self.cl//4, 1 if needs_actions else action_space)
            )
    
    def forward(self, s, action=None):
        graph_embedding = self.graph_head(s)
        if self.needs_actions:
            graph_embedding = torch.cat((graph_embedding, action), -1)
        return self.q_head(graph_embedding)
    

class GraphPolicyNet(nn.Module):
    def __init__(self, graph_head, action_space):
        super().__init__()
        if graph_head is None:
            self.graph_head = GraphQNet()
        else:
            self.graph_head = graph_head
        cl = self.graph_head.cl
        self.p_head = nn.Sequential(
            nn.Linear(2 * cl, cl),
            nn.ReLU(),
            nn.Linear(cl, cl//2),
            nn.ReLU(),
            nn.Linear(cl//2, cl//4),
            nn.ReLU(),
            nn.Linear(cl//4, action_space),
            nn.Softmax(-1)
            )
        self.action_space = action_space
    
    def forward(self, s):
        action_probs = self.p_head(self.graph_head(s))
        action_probs = flatten_probs(action_probs)
        dist = torch.distributions.Categorical(probs = action_probs)
        return dist, action_probs, action_probs.log()

    def sample(self, s):
        dist, probs, log_probs = self(s)
        return dist.sample(), probs, log_probs


class LinearPolicyNet(nn.Module):

    def __init__(self, latent_size, action_space):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(-1)
        )
        self.action_space = action_space

    def forward(self, x):
        action_probs = self.encoder(x)
        action_probs = flatten_probs(action_probs)
        dist = torch.distributions.Categorical(probs = action_probs)
        return dist, action_probs, action_probs.log()

    def sample(self, s):
        dist, probs, log_probs = self(s)
        return dist.sample(), probs, log_probs


class LinearQNet(AbstractQNet):

    def __init__(self, latent_size, action_space):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        self.action_space = action_space

    def forward(self, x):
        return self.encoder(x)


class ImageQNet(AbstractQNet):

    def __init__(self, shape, action_space):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        c = int(((shape[0] / (2 ** 3)) ** 2) * 64)
        self.linear = nn.Sequential(
            nn.Linear(c, 64),
            nn.Linear(64, 32),
            nn.Linear(32, action_space),
        )
        self.action_space = action_space

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1] * shape[2], shape[3], shape[4])
        enc = self.encoder(x)
        return self.linear(enc)
