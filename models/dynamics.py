"""Contains dynamics model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.core.debugger import set_trace

class Dynamics(nn.Module):
    """Graph Neural Network Dynamics Model."""

    def __init__(self, config, monet_config, stove_config, seed=None):
        """Set up model."""
        super().__init__()
        self.c = config
        # set by calling trainer instance
        self.step_counter = 0

        # export some properties for debugging
        self.prop_dict = {}

        # Interaction Net Core Modules
        cl = stove_config.cl
        enc_input_size = cl

        self.num_obj = monet_config.num_slots
        self.num_dynamics = config.num_dynamics

        self.action_conditioned = stove_config.action_conditioned
        self.action_space = config.action_space
        
        if self.action_conditioned:
        # Action Embedding Layer
            self.n_action_enc = 4
            self.action_embedding_layer = nn.Linear(
                self.action_space, self.num_obj * self.n_action_enc)
            enc_input_size += self.n_action_enc
            # Reward MLP
            self.reward_head0 = nn.Sequential(
                nn.Linear(cl, cl),
                nn.ReLU(),
                nn.Linear(cl, cl),
                )

            self.reward_head1 = nn.Sequential(
                nn.Linear(cl, cl//2),
                nn.ReLU(),
                nn.Linear(cl//2, cl//4),
                nn.ReLU(),
                nn.Linear(cl//4, 1)
                )

        self.state_enc = nn.Linear(enc_input_size, cl)

        # Interaction Net Core Modules

        # softmax assignement module
        self.dyn_inference = nn.Sequential(
            nn.Linear(cl, cl),
            nn.ReLU(),
            nn.Linear(cl, self.num_dynamics)
        )

        # Self-dynamics MLP
        self.self_cores = nn.ModuleList()
        for i in range(self.num_dynamics):
            self.self_cores.append(nn.ModuleList())
            self.self_cores[i].append(nn.Linear(cl, cl))
            self.self_cores[i].append(nn.Linear(cl, cl))

        # Relation MLP
        self.rel_cores = nn.ModuleList()
        for i in range(self.num_dynamics):
            self.rel_cores.append(nn.ModuleList())
            self.rel_cores[i].append(nn.Linear(1 + cl * 2, 2 * cl))
            self.rel_cores[i].append(nn.Linear(2 * cl, cl))
            self.rel_cores[i].append(nn.Linear(cl, cl))

        # Attention MLP
        self.att_net = nn.ModuleList()
        for i in range(self.num_dynamics):
            self.att_net.append(nn.ModuleList())
            self.att_net[i].append(nn.Linear(1 + cl * 2, 2 * cl))
            self.att_net[i].append(nn.Linear(2 * cl, cl))
            self.att_net[i].append(nn.Linear(cl, 1))

        # Affector MLP
        self.affector = nn.ModuleList()
        for i in range(self.num_dynamics):
            self.affector.append(nn.ModuleList())
            self.affector[i].append(nn.Linear(cl, cl))
            self.affector[i].append(nn.Linear(cl, cl))
            self.affector[i].append(nn.Linear(cl, cl))

        # Core output MLP
        # changed this to predict 2 cl output
        self.out = nn.ModuleList()
        for i in range(self.num_dynamics):
            self.out.append(nn.ModuleList())
            self.out[i].append(nn.Linear(cl + cl, cl + cl))
            self.out[i].append(nn.Linear(cl + cl, 2 * cl))

        # Attention mask
        diag_mask = 1 - torch.eye(
            self.num_obj,
            ).unsqueeze(2).unsqueeze(0)
        self.register_buffer('diag_mask', diag_mask)

        if self.c.debug_xavier:
            print('Using xavier init for interaction.')
            self.weight_init()

        self.nonlinear = F.elu if self.c.debug_nonlinear == 'elu' else F.relu

        # generative transition likelihood std
        std = self.c.transition_lik_std
        if len(std) == 4:
            std = std + (cl - 4) * [0.01]
        elif len(std) == cl // 2:
            pass
        else:
            raise ValueError('Specify valid transition_lik_std.')
        std = torch.Tensor([[std]])
        self.register_buffer('transition_lik_std', std)

    def weight_init(self):
        """Try Xavier initialisation for weights."""
        for i in range(3):
            for j in range(2):
                torch.nn.init.xavier_uniform_(self.self_cores[i][j].weight)
                torch.nn.init.constant_(self.self_cores[i][j].bias, 0.1)
                torch.nn.init.xavier_uniform_(self.out[i][j].weight)
                torch.nn.init.constant_(self.out[i][j].bias, 0.1)
            for j in range(3):
                torch.nn.init.xavier_uniform_(self.rel_cores[i][j].weight)
                torch.nn.init.constant_(self.rel_cores[i][j].bias, 0.1)
                torch.nn.init.xavier_uniform_(self.att_net[i][j].weight)
                torch.nn.init.constant_(self.att_net[i][j].bias, 0.1)
                torch.nn.init.xavier_uniform_(self.affector[i][j].weight)
                torch.nn.init.constant_(self.affector[i][j].bias, 0.1)

        torch.nn.init.xavier_uniform_(self.state_enc.weight)
        torch.nn.init.constant_(self.state_enc.bias, 0.1)

        torch.nn.init.xavier_uniform_(self.std_layer1.weight)
        torch.nn.init.xavier_uniform_(self.std_layer2.weight)

        torch.nn.init.constant_(self.std_layer1.bias, 0.1)
        torch.nn.init.constant_(self.std_layer2.bias, 0.1)

    def constrain_z_dyn(self, z, z_std=None):
        """Constrain z parameters as predicted by dynamics network.

        Args:
            z (torch.Tensor), (n, o, cl//2): Per object means from dynamics
                network: (positions, velocities, latents).
            z_std (torch.Tensor), (n, o, cl//2): Per object stds ...

        Returns:
            z_c, z_std (torch.Tensor), (n, o, cl//2): Constrained params.

        """
        # Predict positions as velocity differences.
        # velocities between -1 and 1 also and then with vel boound
        # z_c = torch.cat(
        #     [(2 * torch.sigmoid(z[..., :4]) - 1),
        #      (2 * torch.sigmoid(z[..., 4:]) - 1)
        #      ], -1)
        
        # see if different treatments of different parts might be necessary
        z_c = 2 * torch.sigmoid(z) - 1
        # z_std already has positions at this point
        # then constrain velocities
        # then latents
        if z_std is not None:
            # extra stds for latent dimensions
            z_std = torch.cat(
                [self.c.pos_var * torch.sigmoid(z_std[..., :2]),
                 0.04 * torch.sigmoid(z_std[..., 2:8]),
                 self.c.debug_latent_q_std * torch.sigmoid(z_std[..., 8:])], -1)
            assert torch.all(z_std > 0)
            return z_c, z_std + 1e-8

        else:
            return z_c, None

    def forward(self, s, actions=None):
        """Dynamics core. Predict future state given previous.

        Args:
            s (torch.Tensor), (n, o, cl//2): State code.
            core_idct (int): Currently unused. Option for multiple prediction
                cores, as in VIN.
            actions (torch.Tensor), (n, o): If self.c.action_conditioned,
                contains actions which affect the following state and reward.
            obj_appearances (torch.Tensor), (n, o, 3): Colored appearance info
                helps core infer differences between objects.

        Returns:
            out (torch.Tensor), (n, o, cl): New state code means and stds.
            reward (torch.Tensor), (n,): Predicted rewards.

        """
        # set_trace()

        result = torch.zeros_like(torch.cat([s, s], -1))

        if actions is not None:
            action_embedding = self.action_embedding_layer(actions)
            action_embedding = action_embedding.view(
                [action_embedding.shape[0], -1, self.n_action_enc])

        if actions is not None:
            s = torch.cat([s, action_embedding], -1)

        # add back positions for distance encoding
        s = torch.cat([s[..., :2], self.state_enc(s)[..., 2:]], -1)

        # infer object dynamics type (n, o, cl) -> (n, o, num_dyn)
        dyn_types = F.softmax(self.dyn_inference(s), -1)

        for core_idx in range(self.num_dynamics):
            self_sd_h1 = self.nonlinear(self.self_cores[core_idx][0](s))
            self_dynamic = self.self_cores[core_idx][1](self_sd_h1) + self_sd_h1

            object_arg1 = s.unsqueeze(2).repeat(1, 1, self.num_obj, 1)
            object_arg2 = s.unsqueeze(1).repeat(1, s.shape[1], 1, 1)
            distances = (object_arg1[..., 0] - object_arg2[..., 0])**2 +\
                        (object_arg1[..., 1] - object_arg2[..., 1])**2
            distances = distances.unsqueeze(-1)

            # shape (n, o, o, 2cl+1)
            combinations = torch.cat((object_arg1, object_arg2, distances), 3)
            rel_sd_h1 = self.nonlinear(self.rel_cores[core_idx][0](combinations))
            rel_sd_h2 = self.nonlinear(self.rel_cores[core_idx][1](rel_sd_h1))
            rel_factors = self.rel_cores[core_idx][2](rel_sd_h2) + rel_sd_h2

            attention = self.nonlinear(self.att_net[core_idx][0](combinations))
            attention = self.nonlinear(self.att_net[core_idx][1](attention))
            # change this to sigmoid for saving the size
            attention = torch.sigmoid(self.att_net[core_idx][2](attention))

            # mask out object interacting with itself (n, o, o, cl)
            rel_factors = rel_factors * self.diag_mask * attention

            # relational dynamics per object, (n, o, cl)
            rel_dynamic = torch.sum(rel_factors, 2)

            dynamic_pred = self_dynamic + rel_dynamic

            aff1 = torch.tanh(self.affector[core_idx][0](dynamic_pred))
            aff2 = torch.tanh(self.affector[core_idx][1](aff1)) + aff1
            aff3 = self.affector[core_idx][2](aff2)

            aff_s = torch.cat([aff3, s], 2)
            out1 = torch.tanh(self.out[core_idx][0](aff_s))
            result += (self.out[core_idx][1](out1) + out1) * dyn_types[..., core_idx:core_idx+1]
        assert not torch.any(torch.isnan(result))
        
        if self.action_conditioned:
            # reward prediction
            dynamic_pred_rew = dynamic_pred
            reward_data = self.reward_head0(dynamic_pred_rew)
            # sum to (n, cl)
            reward_data = reward_data.sum(1)
            reward = self.reward_head1(reward_data).view(-1, 1)
            reward = torch.sigmoid(reward)
            return result, reward
        return result
