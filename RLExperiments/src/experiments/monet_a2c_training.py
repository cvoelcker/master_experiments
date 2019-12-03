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


class RLACTrainer(ModelTrainer):
    """
    Handler for a2c training. Code mostly taken from
    https://github.com/cyoon1729/Reinforcement-learning/blob/master/A2C/agent.py and
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    """

    def __init__(self, model, env_name, step_size, optimizer, episodes,
                 trajectory_length, gamma, step_size_decay, num_frames,
                 norm_clip,
                 num_envs=10, monet_tb_name='default'):
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
        self.norm_clip = norm_clip

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

    def train(self, conf,):
        value_coeff = conf.TRAIN_SPEC.value_coeff
        entropy_coeff = conf.TRAIN_SPEC.entropy_coeff
        img_coeff = conf.TRAIN_SPEC.image_coeff
        optimizer = self.get_optimizer()

        env = make_envs(self.env_name,
                        self.num_processes,
                        seed=43,
                        device='cuda',
                        log_dir='../log')
        obs_buffer = []

        obs = env.reset()

        entropy_term = []
        ac_losses = []

        total_num_steps = self.episodes
        runs = (self.episodes // (
                self.trajectory_length * self.num_processes)) + 1

        step_decrease_interval = 50
        step_size_decrease = (self.step_size - self.step_size_decay) / (
                runs / step_decrease_interval)

        log_interval = 50
        log_steps = log_interval * (
                self.trajectory_length * self.num_processes)

        print(f'Training agent on {total_num_steps} steps for {runs}')
        print(
            f'Decreasing lr from {self.step_size} to {self.step_size_decay} '
            f'incrementally by {step_size_decrease} every '
            f'{step_decrease_interval} steps')

        running_rewards = np.array([0.0] * self.num_processes)
        episode_rewards = []
        _log_steps = log_steps
        for e in tqdm(list(range(runs))):
            # generate trajectory
            optimizer.zero_grad()

            log_probs = []
            train_rewards = []
            masks = []
            entropies = []
            values = []
            image_losses = []
            for _ in range(self.trajectory_length):
                actor, critic, loss = self.model.ac_net(
                        obs.cuda(),
                        gradient_through_image=conf.MODEL_SPEC.gradient_through_image,
                        use_image_model=conf.MODEL_SPEC.use_image_model,
                        only_image_model=conf.MODEL_SPEC.only_image_model,
                        replace_relational_with_noise=conf.MODEL_SPEC.replace_relational_with_noise,
                        replace_image_with_noise=conf.MODEL_SPEC.replace_image_with_noise
                        )
                if conf.TRAIN_SPEC.train_object_model:
                    torch.mean(loss * img_coeff/self.trajectory_length).backward(retain_graph=conf.MODEL_SPEC.gradient_through_image)
                image_losses.append(loss)
                action = actor.sample()
                log_prob = actor.log_prob(action)
                log_probs.append(log_prob)

                entropies.append(actor.entropy())

                obs, rewards, done, _ = env.step(action.unsqueeze(1))
                obs_buffer.append(obs)

                train_rewards.append(rewards.squeeze())
                values.append(critic.squeeze())
                masks.append(torch.Tensor(1. - done))

                running_rewards += rewards.detach().cpu().numpy().squeeze()
                if np.any(done):
                    episode_rewards.extend([r for r, d in zip(running_rewards, done) if d])
                    running_rewards *= 1. - done

            # calculate loss update
            image_losses = torch.stack(image_losses, 0).cuda()
            train_rewards = torch.stack(train_rewards, 0).cuda()
            log_probs = torch.stack(log_probs, 0).cuda()
            entropies = torch.stack(entropies, 0).cuda()
            masks = torch.stack(masks, 0).cuda()
            values = torch.stack(values, 0).cuda()
            with torch.no_grad():
                _, next_value, _ = self.model.ac_net(obs.cuda(), gradient_through_image=False)
            qval = next_value.squeeze()
            qvals = torch.zeros((self.trajectory_length, self.num_processes)).cuda()
            for t in reversed(range(self.trajectory_length)):
                qval = train_rewards[t] + self.gamma * qval * masks[t]
                qvals[t] = qval
            # calculate advantage from predicted rewards and actual
            advantage = qvals - values
            # calculate loss functions
            actor_loss = torch.mean(-(log_probs * advantage.detach()))
            critic_loss = torch.mean(advantage.pow(2))
            entropy_loss = torch.mean(entropies)
            image_loss = torch.mean(image_losses)
            
            ac_loss = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy_loss # + img_coeff * image_loss
            ac_loss.backward()
            nn.utils.clip_grad_norm_(list(self.model.ac_net.parameters()), self.norm_clip)
            optimizer.step()
            self.model.ac_net.reset_loss()

            # update the monet model

            ac_losses.append(ac_loss.detach().cpu().numpy())
            entropy_term.append(entropies.detach().cpu().numpy())

            if (e + 1) % log_interval == 0:
                if len(episode_rewards) > 0:
                    # rewards, eval_steps = evaluate_model(self.model, env)
                    handler_data = {
                        'tf_logging': {'ac_loss': np.mean(ac_losses),
                                       'entropy': np.mean(entropy_term),
                                       'max_reward': np.max(episode_rewards),
                                       'mean_reward': np.mean(episode_rewards),
                                       },
                        'step': _log_steps}
                    self.notify_handlers(handler_data)
                    entropy_term = []
                    ac_losses = []
                    # train_monet(self.model.ac_net.object_detection, obs_buffer, self.monet_logger, e)
                    obs_buffer = []
                    episode_rewards = []
                    _log_steps = log_steps
                else:
                    _log_steps += log_steps

            if (e + 1) % step_decrease_interval == 0:
                self.update_step_size(
                    self.step_size - step_size_decrease)
                optimizer = self.get_optimizer()

        env.close()
        self.reset_handlers()


def train_monet(model, obs, handler, e):
    obs = torch.cat([o.view(-1, 3, 128, 128).cpu() / 255. for o in obs])[:8 * 500]
    trainloader = torch.utils.data.DataLoader(obs,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=8)
    train_util.run_training(
            model, trainloader, num_epochs=3, initialize=False, parallel=True,
            batch_size=8, anomaly_testing=False, norm_clip=50,
            beta_overwrite=1., tbhandler=handler)
    del(trainloader)
