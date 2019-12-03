from abc import ABC, abstractmethod
import warnings
import sys

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from scipy.special import expit
from spatial_monet.util import experiment_config, train_util

from src.rl_algorithms.agents import Agent
from src.experiments.model_evaluation import evaluate_model
from src.util.env_util import make_envs
from src.experiments.handlers import PrintHandler, TensorboardHandler



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


class RLPPOTrainer(ModelTrainer):
    """
    Handler for a2c training. Code mostly taken from
    https://github.com/cyoon1729/Reinforcement-learning/blob/master/A2C/agent.py and
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    """

    def __init__(self, model, env_name, step_size, optimizer, episodes,
                 trajectory_length, gamma, step_size_decay, num_frames,
                 ppo_epochs, batch_size,
                 clip_param, value_loss_coeff, entropy_coeff,
                 image_loss_coeff, max_grad_norm, 
                 num_envs=10, monet_tb_name='default'):
        assert isinstance(model, Agent), "Reinforcement training expect " \
                                         "agent models"
        self.seed = 1337

        self.episodes = episodes
        self.env_name = env_name
        self.trajectory_length = trajectory_length
        self.num_processes = num_envs
        self.gamma = gamma
        self.step_size = step_size
        self.step_size_decay = step_size_decay
        self.num_frames = num_frames

        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = (self.trajectory_length * self.num_processes) // self.batch_size
        self.clip_param = clip_param
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.image_loss_coeff = image_loss_coeff
        self.max_grad_norm = max_grad_norm

        # if issubclass(optimizer, Optimizer):
        self.optimizer_class = optimizer
        self.optimizer = None
        self.optimizer_initialized = False
        # else:
        #     raise NotImplementedError('Cannot instantiate non optimizer')
        self.monet_logger = TensorboardHandler(reset_logdir=True, logdir='../logs/' + monet_tb_name)
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

    def train(self, conf):
        optimizer = self.get_optimizer()

        env = make_envs(self.env_name,
                        self.num_processes,
                        seed=43,
                        device='cuda',
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

        log_interval = 10
        log_steps = log_interval * (
                self.trajectory_length * self.num_processes)

        print(f'Training agent on {total_num_steps} steps for {runs}')
        print(
            f'Decreasing lr from {self.step_size} to {self.step_size_decay} '
            f'incrementally by {step_size_decrease} every '
            f'{step_decrease_interval} steps')

        running_rewards = np.array([0.0] * self.num_processes)
        print(obs.shape[1:])
        buffer = Buffer(self.trajectory_length, self.num_processes, obs.shape[1:], 'cuda')
        episode_rewards = []
        for e in tqdm(list(range(runs))):
            # generate trajectory
            entropies = []
            with torch.no_grad():
                for _ in range(self.trajectory_length):
                    actor, critic, _ = self.model.ac_net(
                            obs.cuda(),
                            gradient_through_image=False,
                            use_image_model=conf.MODEL_SPEC.use_image_model,
                            only_image_model=conf.MODEL_SPEC.only_image_model,
                            replace_relational_with_noise=conf.MODEL_SPEC.replace_relational_with_noise,
                            replace_image_with_noise=conf.MODEL_SPEC.replace_image_with_noise
                            )
                    action = actor.sample()
                    entropies.append(actor.entropy().detach().cpu().numpy())
                    log_prob = actor.log_prob(action).detach()
                    n_obs, rewards, done, _ = env.step(action.unsqueeze(1))
                    running_rewards += rewards.detach().cpu().numpy().squeeze()
                    buffer.insert(log_prob, action, obs, critic, rewards, done)
                    obs = n_obs

                    if np.any(done):
                        episode_rewards.extend([r for r, d in zip(running_rewards, done) if d])
                        running_rewards *= 1. - done

                _, next_v, _ = self.model.ac_net(
                    obs.cuda(),
                    gradient_through_image=False,
                    use_image_model=conf.MODEL_SPEC.use_image_model,
                    only_image_model=conf.MODEL_SPEC.only_image_model,
                    replace_relational_with_noise=conf.MODEL_SPEC.replace_relational_with_noise,
                    replace_image_with_noise=conf.MODEL_SPEC.replace_image_with_noise
                )
                buffer.calc_rewards(next_v, self.gamma)

            # PPO STYLE
            optimizer = self.get_optimizer()
            for ppo_epoch in range(self.ppo_epochs):
                permutation = np.random.permutation(self.trajectory_length * self.num_processes)
                for batch in range(self.num_batches):
                    perm = permutation[batch*self.batch_size:(1+batch)*self.batch_size]
                    prob_old, acts, ad, _obs, qvals, v_old = buffer.get_random(perm)
                    
                    actor, new_values, img_loss = self.model.ac_net(
                        _obs.cuda(),
                        gradient_through_image=conf.MODEL_SPEC.gradient_through_image,
                        use_image_model=conf.MODEL_SPEC.use_image_model,
                        only_image_model=conf.MODEL_SPEC.only_image_model,
                        replace_relational_with_noise=conf.MODEL_SPEC.replace_relational_with_noise,
                        replace_image_with_noise=conf.MODEL_SPEC.replace_image_with_noise
                        )

                    # get current entropy
                    dist_entropy = torch.mean(actor.entropy())

                    # clipped action loss
                    new_action_log_prob = actor.log_prob(acts)
                    r = torch.exp(new_action_log_prob - prob_old)
                    clamped_r = torch.clamp(r, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    surr1 = r * ad
                    surr2 = clamped_r * ad
                    action_loss = -torch.min(surr1, surr2).mean()
                    
                    # calculate clipped value loss
                    value_clipped = (new_values - v_old).clamp(-self.clip_param, self.clip_param)
                    value_pred_clipped = v_old + value_clipped
                    value_losses = (new_values - qvals).pow(2)
                    value_losses_clipped = (value_pred_clipped - qvals).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    # value_loss = (new_values - qvals).pow(2).mean()

                    # collect losses
                    optimizer.zero_grad()
                    ppo_loss = action_loss + \
                        self.value_loss_coeff * value_loss - \
                        self.entropy_coeff * dist_entropy
                    
                    full_loss = ppo_loss + self.image_loss_coeff * img_loss.mean()
                    full_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.ac_net.parameters(), self.max_grad_norm)
                    optimizer.step()
            buffer.clear()
            entropy_term.append(np.mean(entropies))

            if (e + 1) % log_interval == 0:
                # rewards, eval_steps = evaluate_model(self.model, env)
                handler_data = {
                    'tf_logging': {# 'ac_loss': np.mean(ac_losses),
                                   'entropy': np.mean(entropy_term),
                                   'max_reward': np.max(episode_rewards),
                                   'mean_reward': np.mean(episode_rewards),
                                   },
                    'step': log_steps}
                self.notify_handlers(handler_data)
                entropy_term = []
                ac_losses = []
                # train_monet(self.model.ac_net.object_detection, obs_buffer, self.monet_logger, e)
                obs_buffer = []
                # running_rewards = np.array([0.0] * self.num_processes)
                episode_rewards = []

            if (e + 1) % step_decrease_interval == 0:
                self.update_step_size(
                    self.step_size - step_size_decrease)
                optimizer = self.get_optimizer()

        env.close()
        self.reset_handlers()


class Buffer():
    def __init__(self, trajectory_len, num_envs, image_shape, device):
        self.trajectory_length = trajectory_len
        self.num_envs = num_envs
        self.image_shape = image_shape
        self.device = device
        self.log_probs = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.actions = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.advantage = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.obs_buffer = torch.zeros(trajectory_len, num_envs, *image_shape, requires_grad=False).cpu()
        self.qvals = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.values = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.rewards = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.masks = torch.zeros(trajectory_len, num_envs, requires_grad=False).cpu()
        self.counter = 0
        self.is_flat = False

    def insert(self, log_probs, actions, obs, values, reward, done):
        self.log_probs[self.counter] = log_probs
        self.actions[self.counter] = actions
        self.obs_buffer[self.counter] = obs
        self.values[self.counter] = values.squeeze()
        self.rewards[self.counter] = reward.squeeze()
        self.masks[self.counter] = torch.tensor(1. - done).to(self.device)
        self.counter += 1
    
    def clear(self):
        self.is_flat = False
        self.log_probs[:] = 0.
        self.actions[:] = 0.
        self.advantage[:] = 0.
        self.obs_buffer[:] = 0.
        self.qvals[:] = 0.
        self.values[:] = 0.
        self.rewards[:] = 0.
        self.masks[:] = 0.
        self.counter = 0

    def flat(self):
        self.is_flat = True
        self.l = self.log_probs.flatten(end_dim=1)
        self.ac = self.actions.flatten(end_dim=1)
        self.ad = self.advantage.flatten(end_dim=1)
        self.o = self.obs_buffer.flatten(end_dim=1)
        self.q = self.qvals.flatten(end_dim=1)
        self.v = self.values.flatten(end_dim=1)

    def get_random(self, perm):
        d = self.device
        if not self.is_flat:
            self.flat()
        return self.l[perm].to(d), self.ac[perm].to(d), self.ad[perm].to(d), self.o[perm].to(d), self.q[perm].to(d), self.v[perm].to(d)

    def calc_rewards(self, next_v, gamma):
        qval = next_v.squeeze()
        for t in reversed(range(self.trajectory_length)):
            qval = self.rewards[t] + gamma * qval * self.masks[t]
            self.qvals[t] = qval
        # calculate advantage from predicted rewards and actual
        self.advantage = (self.qvals - self.values).detach()
