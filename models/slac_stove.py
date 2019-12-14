import util.buffer as buffer
import copy

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import pickle

class SLACAgent():

    def __init__(self, policy, q_1, q_2, model, rl_lr=0.0001, latent_lr=0.0001,
                 update_steps_latent=10, update_steps_rl=10, update_target_steps=5,
                 grad_clip_model=50, grad_clip_rl=20, target_entropies=None, rl_gamma = 0.99, 
                 initial_alpha = 0.01, target_gamma=0.995, debug=False, **kwargs):
        self.policy = policy
        self.q_1 = q_1
        self.q_2 = q_2
        self.q_1_target = self.copy_q(q_1)
        self.q_2_target = self.copy_q(q_2)
        self.model = model
        if target_entropies is None:
            self.target_entropies = torch.tensor(-0.98 * np.log(1/self.q_1.action_space)).cuda()
        else:
            self.target_entropies = torch.tensor(target_entropies).cuda()
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device='cuda')
        self.alpha = self.log_alpha.exp()

        self.optim_model = Adam(model.parameters(), latent_lr)
        self.optim_p = Adam(policy.parameters(),rl_lr)
        self.optim_q_1 = Adam(q_1.parameters(), rl_lr)
        self.optim_q_2 = Adam(q_2.parameters(), rl_lr)
        self.optim_e = Adam([self.log_alpha], rl_lr)

        self.update_steps_latent = update_steps_latent
        self.update_steps_rl = update_steps_rl
        self.update_target_steps = update_target_steps

        self.grad_clip_model = grad_clip_model
        self.grad_clip_rl = grad_clip_rl

        self.gamma = rl_gamma
        self.target_gamma = target_gamma

        self.debug = debug
        

    def update(self, batch):
        m_l = []
        for i in range(self.update_steps_latent):
            l = self.update_model(batch)
            m_l.append(l)
        q1_l = []
        q2_l = []
        p_l  = []
        e_l  = []
        ent  = []
        for i in range(self.update_steps_rl):
            q1, q2, p, e, er = self.update_rl(batch)
            q1_l.append(q1)
            q2_l.append(q2)
            p_l.append(p)
            e_l.append(e)
            ent.append(er)
            if i % self.update_target_steps == 0:
                self.update_target_networks()
        q1_l = torch.stack(q1_l, 0)
        q2_l = torch.stack(q2_l, 0)
        p_l  = torch.stack(p_l, 0)
        e_l  = torch.stack(e_l, 0)
        m_l  = torch.stack(m_l, 0)
        ent  = torch.stack(ent, 0)
        return {'q1': torch.mean(q1_l),
                'q2': torch.mean(q2_l),
                'p': torch.mean(p_l),
                'e': torch.mean(e_l),
                'm': torch.mean(m_l),
                'ent': torch.mean(ent)}
    
    def update_model(self, batch):
        obs = batch['x']
        actions = batch['a']
        mask = torch.cumsum(batch['d'], -1)
        loss, _, _ = self.model(obs, actions=actions, mask=mask)
        (-1*loss).mean().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_model)
        self.optim_model.step()
        return loss.detach()

    def update_rl(self, batch):
        """
        Gathers the latent inference for RL, the q and policy losses and performsthe update step
        """
        if self.debug:
            from IPython.core.debugger import Tracer
            Tracer()()
        with torch.no_grad():
            obs = batch['x']
            actions = batch['a']
            all_latents = self.model.infer_latent(obs, actions).cuda()
            # all_latents = self.get_latent(batch, pred=False) # final latent is only inferred? TODO Test both
            obs_dump = obs[:, -2]
            latents = all_latents[:, -2]
            # pickle.dump(obs_dump.cpu().detach().numpy(), open('dump_obs.pkl', 'wb'))
            # pickle.dump(latents.cpu().detach().numpy(), open('dump_lat.pkl', 'wb'))
            # recon = self.model.img_model.reconstruct_from_latent(latents)
            # pickle.dump(recon.cpu().detach().numpy(), open('dump_recon.pkl', 'wb'))
            # exit()
            next_latents = all_latents[:, -1]
            actions = batch['a'][:, -2] # second to last action leads to last state
            rewards = batch['r'][:, -2] # potentially sample reward from reward model
            dones = batch['d'][:, -2]
        
        # get the rl losses
        q1_loss, q2_loss = self.calc_critic_loss(latents, next_latents, batch['a_idx'][:, -2:-1], rewards, dones)
        policy_loss, entropy_loss, mean_entropies = self.calc_policy_loss(latents)
        # print(q1_loss)
        # print(q2_loss)
        # print(policy_loss)
        # print(entropy_loss)

        # run optimizers (TODO: check if gradients in latents are nott interferring)
        self.optim_step(self.optim_q_1, q1_loss, self.q_1, self.grad_clip_rl, retain=True)
        self.optim_step(self.optim_q_2, q2_loss, self.q_2, self.grad_clip_rl, retain=True)
        self.optim_step(self.optim_p, policy_loss, self.policy, self.grad_clip_rl, retain=False)
        self.optim_step(self.optim_e, entropy_loss, self.alpha, 0, no_clip=True, retain=False)
        self.alpha = self.log_alpha.exp()

        return q1_loss.detach(), q2_loss.detach(), policy_loss.detach(), self.alpha.detach(), -mean_entropies
        # print(self.alpha)

    def optim_step(self, optim, loss, net, clip, no_clip=False, retain=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain)
        if not no_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()
        optim.zero_grad()
        
    def calc_critic_loss(self, latents, next_latents, actions, rewards, dones):
        with torch.no_grad():
            # calculate one step lookahead of policy
            _, probs, entropies = self.policy.sample(next_latents)
            next_q1 = self.q_1_target(next_latents)
            next_q2 = self.q_2_target(next_latents)
            next_q = probs * (torch.min(next_q1, next_q2) - self.alpha * probs.log())
            next_q = torch.sum(next_q, -1)
            target_q = (rewards + self.gamma * next_q)

        curr_q1 = self.q_1(latents)
        curr_q1 = curr_q1.gather(-1, actions) # get the correct part of the multi action q function
        curr_q1 = curr_q1.view(-1)
        curr_q2 = self.q_2(latents)
        curr_q2 = curr_q2.gather(-1, actions)
        curr_q2 = curr_q2.view(-1)

        q1_loss = torch.mean((curr_q1 - target_q.detach()).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q.detach()).pow(2))

        return q1_loss, q2_loss

    def calc_policy_loss(self, latents):
        _, probs, entropies = self.policy(latents)
        with torch.no_grad():
            q1 = self.q_1(latents)
            q2 = self.q_2(latents)
            q = torch.min(q1, q2)
        # Policy objective is maximization of (-Q + alpha * probs).
        # print(probs[-1])
        # print(q[-1])
        # print((probs * (self.alpha * probs.log() - q))[-1])
        # print(self.alpha)
        policy_loss = torch.mean(torch.sum(probs * (self.alpha * probs.log() - q), -1))
        entropy_loss, mean_entropies = self.calc_entropy_loss(probs.detach())
        return policy_loss, entropy_loss, mean_entropies

    def calc_entropy_loss(self, probs):
        """
        calculates the alpha parameter of the entropy
        """
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        # print(self.target_entropies)
        entropy = torch.sum(probs * probs.log(), -1)
        # print(f'Entropy {torch.mean(entropy)} vs target {self.target_entropies}')
        entropy_loss = -torch.mean(self.log_alpha * (entropy + self.target_entropies))
        return entropy_loss, entropy

    def update_target_networks(self):
        """
        Updates the target q function networks with a moving average of params
        """
        def softupdate(param1, param2):
            return (self.target_gamma * param1.data) + ((1 - self.target_gamma) * param2.data)
        with torch.no_grad():
            for param_target, param in zip(self.q_1_target.parameters(), self.q_1.parameters()):
                param_target.data.copy_(softupdate(param_target, param))
            for param_target, param in zip(self.q_2_target.parameters(), self.q_2.parameters()):
                param_target.data.copy_(softupdate(param_target, param))

    def get_latents(self, batch, pred = False):
        latents = self.model.infer_latent(batch)
        if pred:
            pred = self.model.rollout(latents[-1], num=1)
            latents = torch.cat((latents, pred), 1)
        return latents

    def copy_q(self, net):
        copy_net = copy.deepcopy(net)
        for p in copy_net.parameters():
            p.requires_grad = False
        return copy_net

    
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


class GraphQNet(nn.Module):
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
        min_probs = torch.ones_like(action_probs) * 0.001 / self.action_space
        action_probs = min_probs + 0.999 * action_probs
        dist = torch.distributions.Categorical(probs = action_probs)
        return dist, action_probs, dist.entropy()

    def sample(self, s):
        dist, probs, entropy = self(s)
        return dist.sample(), probs, entropy


class ImagePolicyNet(nn.Module):
    def __init__(self, img_size, action_space, latent_size=256):
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        image_output = 64 * (img_size.shape[1]/8) * (img_size.shape[1]/8)
        self.recurrent = nn.GRUCell(image_output + action_space, latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size//2),
            nn.ReLU(),
            nn.Linear(latent_size//2, latent_size//4),
            nn.ReLU(),
            nn.Linear(latent_size//4, action_space)
        )
        hidden_prior = torch.distributions.Normal(0., 0.01)
        self.latent_size = latent_size

    def forward(self, image, action, hidden):
        batch = image.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch)
        T = image.shape[1]
        for t in range(T):
            image_embedding = self.image_encoder(image[:, t])
            lstm_input = torch.cat((image_embedding, action[:, t]), -1)
            hidden = self.recurrent(lstm_input, hidden)
        action_probs = self.mlp(hidden)
        # hard min to prevent entropy collapse
        min_probs = torch.ones_like(action_probs) * 0.01
        action_probs = min_probs + 0.99 * action_probs
        dist = torch.distributions.Categorical(probs = action_probs)
        return dist, action_probs, dist.entropy()

    def sample(self, s, eval=False):
        dist, probs, entropy = self(s)
        if eval:
            return dist.sample(), probs, entropy
        else:
            return dist.sample(), probs, entropy
    
    def init_hidden(self, batch):
        return hidden_prior.sample((batch, self.latent_size))


class VariationalTimeSeriesNet(nn.Module):
    def __init__(self, input_shape, ):

        # latent encoder


        # latent decoder


        # latent propagation component


        # priors

        pass
    
    def forward(self, x, a, r):
        pass
    
    def calc_loss(self):
        pass

    def combine_latent(self, mu_im, std_im, mu_dyn, std_dyn):
        pass