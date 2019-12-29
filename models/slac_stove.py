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
    
    def update_model(self, batch, pretrain_img=False):
        obs = batch['X']
        actions = batch['a']
        rewards = batch['r']
        mask = torch.cumsum(batch['d'], -1)
        loss, _, _ = self.model(obs, actions, rewards, mask, pretrain=pretrain_img)
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
            obs = batch['X']
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
        # self.optim_step(self.optim_e, entropy_loss, self.alpha, 0, no_clip=True, retain=False)
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
            _, probs, _ = self.q_1.sample_action(next_latents)
            next_q1 = self.q_1_target(next_latents)
            next_q2 = self.q_2_target(next_latents)
            next_q = probs * (torch.min(next_q1, next_q2) - self.alpha * probs.log())
            # old update
            # next_q = torch.log(torch.sum(trch.exp(next_q, -1)))
            # soft q update
            next_q = torch.log(torch.sum(torch.exp(next_q, -1)))
            target_q = (rewards + (1. - dones) * self.gamma * next_q)

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
        _, probs, _ = self.q_1.get_probs(latents)
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


class SACAgent():

    def __init__(self, policy, q_1, q_2, model, rl_lr=0.0001, latent_lr=0.0001,
                 update_steps_latent=10, update_steps_rl=10, update_target_steps=5,
                 grad_clip_model=50, grad_clip_rl=20, target_entropies=None, rl_gamma = 0.99, 
                 initial_alpha = 0.01, target_gamma=0.995, debug=False, **kwargs):
        self.q_1 = q_1
        self.q_2 = q_2
        self.q_1_target = self.copy_q(q_1)
        self.q_2_target = self.copy_q(q_2)
        if target_entropies is None:
            self.target_entropies = torch.tensor(-0.98 * np.log(1/self.q_1.action_space)).cuda()
        else:
            self.target_entropies = torch.tensor(target_entropies).cuda()
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device='cuda')
        self.alpha = self.log_alpha.exp()

        self.optim_q_1 = Adam(q_1.parameters(), rl_lr)
        self.optim_q_2 = Adam(q_2.parameters(), rl_lr)
        self.optim_e = Adam([self.log_alpha], rl_lr)

        self.update_steps_rl = update_steps_rl
        self.update_target_steps = update_target_steps

        self.grad_clip_rl = grad_clip_rl

        self.gamma = rl_gamma
        self.target_gamma = target_gamma

        self.debug = debug

    def update(self, batch):
        q1_l = []
        q2_l = []
        e_l  = []
        ent  = []
        for i in range(self.update_steps_rl):
            q1, q2, e, er = self.update_rl(batch)
            q1_l.append(q1)
            q2_l.append(q2)
            e_l.append(e)
            ent.append(er)
            if i % self.update_target_steps == 0:
                self.update_target_networks()
        q1_l = torch.stack(q1_l, 0)
        q2_l = torch.stack(q2_l, 0)
        e_l  = torch.stack(e_l, 0)
        ent  = torch.stack(ent, 0)
        return {'q1': torch.mean(q1_l),
                'q2': torch.mean(q2_l),
                'e': torch.mean(e_l),
                'ent': torch.mean(ent)}
    
    def update_rl(self, batch):
        """
        Gathers the latent inference for RL, the q and policy losses and performsthe update step
        """
        if self.debug:
            from IPython.core.debugger import Tracer
            Tracer()()
        with torch.no_grad():
            obs = batch['X'][:, -5:-1]
            next_obs = batch['X'][:, -4:]
            actions = batch['a_idx'][:, -2:-1] # second to last action leads to last state
            rewards = batch['r'][:, -2] # potentially sample reward from reward model
            dones = batch['d'][:, -2]
        
        # get the rl losses
        q1_loss, q2_loss, mean_entropies = self.calc_critic_loss(obs, next_obs, actions, rewards, dones)
        
        # run optimizers (TODO: check if gradients in latents are not interferring)
        self.optim_step(self.optim_q_1, q1_loss, self.q_1, self.grad_clip_rl, retain=True)
        self.optim_step(self.optim_q_2, q2_loss, self.q_2, self.grad_clip_rl, retain=True)
        # self.optim_step(self.optim_e, entropy_loss, self.alpha, 0, no_clip=True, retain=False)
        self.alpha = self.log_alpha.exp()

        return q1_loss.detach(), q2_loss.detach(), self.alpha.detach(), -mean_entropies
        # print(self.alpha)

    def optim_step(self, optim, loss, net, clip, no_clip=False, retain=False):
        optim.zero_grad()
        loss.backward(retain_graph=retain)
        if not no_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()
        optim.zero_grad()
        
    def calc_critic_loss(self, obs, next_obs, actions, rewards, dones):
        with torch.no_grad():
            # calculate one step lookahead of policy
            _, probs, _ = self.q_1.sample_action(next_obs)
            next_q1 = self.q_1_target(next_obs)
            next_q2 = self.q_2_target(next_obs)
            next_q = probs * (torch.min(next_q1, next_q2) - self.alpha * probs.log())
            # old update
            # next_q = torch.sum(next_q, -1)
            # soft q update
            next_q = torch.log(torch.sum(torch.exp(next_q, -1)))
            target_q = (rewards + (1. - dones) * self.gamma * next_q)
            mean_entropies = torch.mean(torch.sum(probs * probs.log(), -1))

        curr_q1 = self.q_1(obs)
        curr_q1 = curr_q1.gather(-1, actions) # get the correct part of the multi action q function
        curr_q1 = curr_q1.view(-1)
        curr_q2 = self.q_2(obs)
        curr_q2 = curr_q2.gather(-1, actions)
        curr_q2 = curr_q2.view(-1)

        q1_loss = torch.mean((curr_q1 - target_q.detach()).pow(2))
        q2_loss = torch.mean((curr_q2 - target_q.detach()).pow(2))

        return q1_loss, q2_loss, mean_entropies

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

    def copy_q(self, net):
        copy_net = copy.deepcopy(net)
        for p in copy_net.parameters():
            p.requires_grad = False
        return copy_net