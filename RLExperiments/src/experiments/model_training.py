from abc import ABC, abstractmethod
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from src.rl_algorithms.agents import Agent
from src.experiments.model_evaluation import evaluate_model
from src.util.env_util import make_envs


def plot_gradient(params):
    for n, p in params:
        # assert p.grad is not None, f'missing gradient information in {n}'
        if p.grad is not None:
            print(f'Gradient: {n}: {torch.norm(p.grad, 1)}')
        else:
            print(f'No gradient for {n}')
        # print(f'weight: {n}: {p}')


class ModelTrainer(ABC):
    def __init__(self, model, step_strategy):
        self.model = model
        self.step_strategy = step_strategy
        self.handlers = []
        super().__init__()

    def register_handler(self, handler):
        self.handlers.append(handler)

    @abstractmethod
    def train(self, data, epochs):
        pass

    def notify_handlers(self, _data):
        for h in self.handlers:
            h.run(self.model, _data)

    def reset_handlers(self):
        for h in self.handlers:
            h.reset()


class RLACTrainer(ModelTrainer):
    """
    Handler for a2c training. Code mostly taken from
    https://github.com/cyoon1729/Reinforcement-learning/blob/master/A2C/agent.py and
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    """

    def __init__(self, model, env_name, step_size, optimizer, episodes,
                 trajectory_length, gamma, step_size_decay, num_frames,
                 num_envs=10):
        assert isinstance(model, Agent), "Reinforcement training expect " \
                                         "agent models"
        self.episodes = episodes
        self.env_name = env_name
        self.trajectory_length = trajectory_length
        self.gamma = gamma
        self.step_size = step_size
        self.step_size_decay = step_size_decay
        self.num_frames = num_frames

        self.seed = 1337
        self.num_processes = num_envs

        # if issubclass(optimizer, Optimizer):
        self.optimizer_class = optimizer
        self.optimizer = None
        self.optimizer_initialized = False
        # else:
        #     raise NotImplementedError('Cannot instantiate non optimizer')
        super().__init__(model, optimizer)

    def get_optimizer(self):
        if not self.optimizer_initialized:
            self.optimizer_initialized = True
            self.optimizer = self.optimizer_class(
                self.model.ac_net.parameters(), self.step_size)
        return self.optimizer

    def update_step_size(self, step_size):
        self.step_size = step_size
        self.optimizer_initialized = False

    def train(self, data, epochs):
        if data is not None:
            warnings.warn("Passed non empty data, will be overwritten",
                          RuntimeWarning, stacklevel=2)
        optimizer = self.get_optimizer()

        env = make_envs(self.env_name,
                        self.num_processes,
                        seed=43,
                        device='cpu',
                        log_dir='../log')
        obs = env.reset()

        entropy_term = []
        ac_losses = []

        total_num_steps = self.episodes
        runs = (self.episodes // (
                self.trajectory_length * self.num_processes)) + 1

        step_decrease_interval = 50
        step_size_decrease = (self.step_size - self.step_size_decay) / (
                runs / step_decrease_interval)

        log_interval = 100
        log_steps = log_interval * (
                self.trajectory_length * self.num_processes)

        print(f'Training agent on {total_num_steps} steps for {runs}')
        print(
            f'Decreasing lr from {self.step_size} to {self.step_size_decay} '
            f'incrementally by {step_size_decrease} every '
            f'{step_decrease_interval} steps')

        running_rewards = [0.0] * self.num_processes
        episode_rewards = []
        for e in tqdm(list(range(runs))):
            # generate trajectory

            log_probs = []
            train_rewards = []
            masks = []
            entropies = []
            values = []
            for _ in tqdm(list(range(self.trajectory_length))):
                # done, rewards = self.model.step_no_gradients(env)
                actor, critic = self.model.ac_net(obs)
                action = actor.sample()
                log_prob = actor.log_prob(action)
                log_probs.append(log_prob)

                entropies.append(actor.entropy())

                obs, rewards, done, _ = env.step(action.unsqueeze(1))
                train_rewards.append(rewards.squeeze())
                values.append(critic.squeeze())
                masks.append(torch.Tensor(1. - done))

                running_rewards = [r + new for r, new
                                   in zip(running_rewards, rewards)]
                for i, d in enumerate(done):
                    if d:
                        episode_rewards.append(running_rewards[i])
                        running_rewards[i] = 0

            # calculate loss update
            train_rewards = torch.stack(train_rewards, 0)
            log_probs = torch.stack(log_probs, 0)
            entropies = torch.stack(entropies, 0)
            masks = torch.stack(masks, 0)
            values = torch.stack(values, 0)
            with torch.no_grad():
                _, next_value = self.model.ac_net(obs)
            qval = next_value.squeeze()
            qvals = torch.zeros((self.trajectory_length, self.num_processes))
            for t in reversed(range(self.trajectory_length)):
                qval = train_rewards[t] + self.gamma * qval * masks[t]
                qvals[t] = qval
            # calculate advantage from predicted rewards and actual
            advantage = qvals - values
            # calculate loss functions
            actor_loss = torch.mean(-(log_probs * advantage.detach()))
            critic_loss = torch.mean(advantage.pow(2))
            entropy_loss = torch.mean(entropies)
            ac_loss = actor_loss + \
                      0.5 * critic_loss - \
                      0.01 * entropy_loss

            # ac_loss, entropy, _, _, info = self.model.gather_updates(
            #     env)
            optimizer.zero_grad()
            ac_loss.backward()
            # plot_gradient(self.model.get_named_parameters())
            nn.utils.clip_grad_norm_(list(self.model.ac_net.parameters()), 0.5)
            optimizer.step()

            ac_losses.append(ac_loss.detach().numpy())
            entropy_term.append(entropies.detach().numpy())

            if (e + 1) % log_interval == 0:
                # rewards, eval_steps = evaluate_model(self.model, env)
                if len(episode_rewards) > 0:
                    handler_data = {
                        'tf_logging': {'loss': np.mean(ac_losses),
                                       'entropy': np.mean(entropy_term),
                                       'max_reward': np.max(episode_rewards),
                                       'mean_reward': np.mean(episode_rewards),
                                       },
                        'step': log_steps}
                    self.notify_handlers(handler_data)
                entropy_term = []
                ac_losses = []
                episode_rewards = []

            if (e + 1) % step_decrease_interval == 0:
                self.update_step_size(
                    self.step_size - step_size_decrease)
                optimizer = self.get_optimizer()

        env.close()
        self.reset_handlers()
