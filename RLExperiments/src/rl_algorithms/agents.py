from abc import ABC, abstractmethod

import torch
import numpy as np

from torch.autograd import Variable


class Agent(ABC):

    @abstractmethod
    def eval_step(self, env):
        pass

    @abstractmethod
    def step(self, env):
        pass

    @abstractmethod
    def get_parameters(self):
        pass


class ACAgent(Agent):
    """
    Wrapper class for env interaction functions. Keeps a reference to the
    actual ac net
    """

    def __init__(self, ac_net, action_space, gamma, trajectory_length,
                 num_envs, device='cpu'):
        self.ac_net = ac_net
        self.action_space = action_space
        self.gamma = gamma
        self.trajectory_length = trajectory_length

        # initializes storage tensors
        self.rewards = torch.zeros((num_envs, trajectory_length))
        self.values = torch.zeros((num_envs, trajectory_length))
        self.log_probs = torch.zeros((num_envs, trajectory_length))
        self.entropy_term = torch.zeros((num_envs, trajectory_length))
        self.bad_masks = torch.ones((num_envs, trajectory_length))
        self.states = torch.zeros(
            (num_envs, trajectory_length, *self.ac_net.image_input_shape()))
        self.actions = torch.zeros((num_envs, trajectory_length))
        self.step_counter = 0

        self.num_envs = num_envs

        self.device = device
        self.to()

        super().__init__()

    def get_parameters(self):
        """
        Returns the models parameters
        """
        return self.ac_net.parameters()

    def get_named_parameters(self):
        """
        Returns the models named parameters
        """
        return self.ac_net.named_parameters()

    def step(self, env):
        """
        Calculate a prediction, action and step in the environment
        Saves all tensors for later
        """
        raise NotImplementedError

    def step_no_gradients(self, env, save=True):
        """
        Calculate a prediction, action and step in the environment
        Saves all tensors for later
        """
        state = env.stacked_obs
        state = Variable(state)
        with torch.no_grad():
            policy_dist, _ = self.ac_net(state)
            action = policy_dist.sample().view(-1, 1)
        _, reward, done, infos = env.step(action)
        if save:
            self.actions[:, self.step_counter] = action.squeeze()
            self.rewards[:, self.step_counter] = reward.squeeze()
            self.states[:, self.step_counter] = state
            self.bad_masks[:, self.step_counter] = torch.Tensor(1. - done)
            self.step_counter += 1

        return done, reward

    def eval_step(self, state):
        """
        Calculate a single step forward with no gradient accumulation
        does not actually step in the environment
        """
        with torch.no_grad():
            return self.ac_net.forward(state)

    def gather_updates(self, env):
        """
        Gather saved environments 
        """
        # loading states from step model and computing predictions
        states = self.states.view(-1, *self.ac_net.image_input_shape())
        actor, critic = self.ac_net(states)

        # getting log_probs from policy logits
        log_probs = actor.log_prob(self.actions.view(-1))

        # reshaping all policys again for reward calculation
        log_probs = log_probs.view(self.num_envs, self.trajectory_length)
        values = critic.view(self.num_envs,
                             self.trajectory_length)
        entropy = actor.entropy().view(
            self.num_envs, self.trajectory_length)

        # bootstrapping from current value function prediction
        qvals = torch.zeros((self.num_envs, self.trajectory_length))
        with torch.no_grad():
            _, next_value = self.ac_net(env.stacked_obs)

        # calculate discounted reward wih bootstrapped value prediction
        qval = next_value.squeeze()
        for t in reversed(range(self.trajectory_length)):
            qval = self.rewards[:, t] + \
                   self.gamma * qval * self.bad_masks[:, t]
            qvals[:, t] = qval

        # calculate advantage from predicted rewards and actual
        advantage = qvals - values

        # calculate loss functions
        actor_loss = torch.mean(-(log_probs * advantage.detach()))
        critic_loss = torch.mean(advantage ** 2)
        entropy_loss = torch.mean(entropy)
        ac_loss = actor_loss + \
                  0.5 * critic_loss - \
                  0.01 * entropy_loss
        self.step_counter = 0
        return ac_loss, entropy_loss, actor_loss, critic_loss, torch.sum(
            self.rewards)

    def reset_trajectory(self):
        """
        reset all current buffers to 0
        """
        self.rewards = torch.zeros((self.num_envs, self.trajectory_length))
        self.values = torch.zeros((self.num_envs, self.trajectory_length))
        self.log_probs = torch.zeros((self.num_envs, self.trajectory_length))
        self.entropy_term = torch.zeros(
            (self.num_envs, self.trajectory_length))
        self.states = torch.zeros(
            (self.num_envs, self.trajectory_length,
             *self.ac_net.image_input_shape()))
        self.actions = torch.zeros((self.num_envs, self.trajectory_length))
        self.step_counter = 0
        self.to()

    def to(self, device=None):
        """
        move all current buffers to device
        """
        if device is None:
            device = self.device
        self.ac_net.to(device)
        self.rewards.to(device)
        self.values.to(device)
        self.log_probs.to(device)
        self.entropy_term.to(device)
        self.bad_masks.to(device)
        self.states.to(device)
        self.actions.to(device)
