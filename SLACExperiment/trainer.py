import numpy as np
import torch
import copy

from tqdm import tqdm

from torch_runner.train.base import AbstractTrainer

class RLTrainer(AbstractTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def check_ready(self):
        return True

    def train_step(self):
        pass

    def train(self, total_steps, batch_size, debug=False):
        # self.explore(initial=True)
        pbar = tqdm(total=total_steps)
        i = 0
        steps = 0
        eval_means = []
        eval_vars = []
        all_obses = []
        while steps < total_steps:
            if (steps % self.eval_every == 0) and not debug:
                eval_mean, eval_var, all_obs = self.eval()
                print(f'{eval_mean}, {eval_var}')
                eval_means.append(eval_mean)
                eval_vars.append(eval_vars)
                # all_obses.append(all_obs)
            batch = self.compose_batch(batch_size)
            losses = self.update(batch)
            losses['rolling_reward'] = torch.tensor(self.rolling_reward()).float()
            self.notify_step_handlers(losses)
            self.explore()
            pbar.update(1)
            # print(f'Seen {self.memory.total_samples} steps')
            steps += 1

        return eval_means, eval_vars, all_obses
    
    def update(self, batch):
        return self.model.update(batch)

    def compose_batch(self, batch_size):
        """
        Returns dict of:
            (batch_size, seq_len, ...) for each of obs, actions, rewards, dones
        """
        buffer_size = len(self.memory)
        assert buffer_size >= batch_size, 'Cannot sample {} sequences from short buffer'.format(batch_size)
        samples = np.random.choice(buffer_size, batch_size)
        batch = []
        for s in samples:
            batch.append(self.memory[s])
        batch = {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}
        return batch

    def rolling_reward(self):
        fill = len(self.memory)
        rewards = self.memory.r_buffer[fill-1000:fill]
        return np.sum(rewards)


class MONetTrainer(AbstractTrainer):

    def train_step(self, data, **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        loss, data_dict, r = self.model(data['X'], actions=data['a'].cuda().float(), pretrain=kwargs['pretrain'])
        self.optimizer.zero_grad()
        torch.mean(-1. * loss).backward()
        if self.clip_gradient:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient_value)
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                # print(f'{n}: {p.grad.max()} {p.grad.min()} {p.grad.mean()}')
                if not torch.isfinite(p.grad.mean()):
                    from IPython.core.debugger import Tracer
                    Tracer()()
        self.optimizer.step()
        return data_dict

    def check_ready(self):
        if self.train_dataloader is None:
            return False
        if self.optimizer is None:
            return False
        if self.model is None:
            return False
        return True

    def append_epoch_info_dict(self, epoch_info_dict, data_dict):
        return data_dict

    def compile_epoch_info_dict(self, data_dict, epoch, **kwargs):
        if 'pretrain' in kwargs:
            self.model.img_model.beta = 1 / (1 + np.exp(-epoch))
        return {'model_state': self.model.state_dict(),}
                #'imgs': (data_dict['imgs'] * 255).type_as(torch.ByteTensor())}


class MONetTester(AbstractTrainer):

    def train_step(self, data): 
        with torch.no_grad():
        # torch.autograd.set_detect_anomaly(True)
            loss, data_dict, r = self.model(data['X'], actions=data['a'].cuda().float())
            z_full, imgs, r = self.model.rollout(
                    data_dict['z_s'][:, -1].cuda().float(),
                    num = data['action'].shape[1],
                    actions=data['action'].cuda().float(),
                    return_imgs=True)
            return {'z': z_full, 'r': r, 'imgs': (imgs * 255).type_as(torch.ByteTensor())}

    def check_ready(self):
        if self.train_dataloader is None:
            return False
        if self.optimizer is None:
            return False
        if self.model is None:
            return False
        return True

    def compile_epoch_info_dict(self, data_dict, e, **kwargs):
        return {'model_state': self.model.state_dict()}


class SLACTrainer(RLTrainer):

    def __init__(self, config, model, env, memory, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.memory = memory
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.done = False
        self.obs, self.latents = self.init_latent(self.env)
        self.eval_epochs = config.eval_epochs
        self.eval_every = 2500

        self.exploration_steps = config.exploration_steps
    
    def explore(self, initial = False):
        with torch.no_grad():
            steps = self.exploration_steps if not initial else 1000
            obs = []
            actions = []
            r_s = []
            d_s = []
            i = 0
            while i < steps:
                i += 1
                obs.append(self.obs)
                action, _, _ = self.model.q_1.sample_action(self.latents.cuda())
                action = action.detach().cpu().numpy()
                actions.append(action)
                self.obs, r, d, _ = self.env.step(action)
                r_s.append(r)
                d_s.append(d)
                if d:
                    i += 4
                    self.obs, self.latents = self.init_latent(self.env)
                else:
                    _obs = self.memory.resize(self.obs)
                    _obs = self.memory.torch_transform(_obs[np.newaxis, :]).unsqueeze(0)
                    action = self.memory.torch_transform(action[np.newaxis, :], action_expand=True)
                    self.latents = self.model.model.update_latent(_obs, action, self.latents.cuda())
        obs = np.stack(obs, 0)
        actions = np.stack(actions, 0).squeeze(-1)
        r_s = np.stack(r_s, 0)
        d_s = 0. + np.stack(d_s, 0)
        self.memory.put(obs, actions, r_s, d_s)
    
    def init_latent(self, env, write=True):
        last_obs = env.reset()
        initial_sample_obs = []
        initial_sample_a = []
        for _ in range(4):
            initial_sample_obs.append(self.memory.resize(last_obs))
            action = env.action_space.sample()
            obs, r, d, _ = env.step(action)
            initial_sample_a.append(action)
            if write:
                self.memory.put(last_obs[np.newaxis, :], 
                                np.array([[action]], dtype=np.uint8), 
                                np.array([[r]], dtype=np.uint8), 
                                np.array([[d]], dtype=np.uint8))
            last_obs = obs
        
        # torchify
        obs = self.memory.torch_transform(np.stack(initial_sample_obs, 0)[np.newaxis, :])
        a = self.memory.torch_transform(np.stack(initial_sample_a, 0)[np.newaxis, :], action_expand=True)
        latent = self.model.model.infer_latent(obs, a)
        # only return final latent
        return last_obs, latent[:, -1]

    def eval(self):
        full_model_state = {
            'model_state': {
                # 'policy': self.model.policy.state_dict(),
                'q1': self.model.q_1.state_dict(),
                'q2': self.model.q_2.state_dict(),
                'q1_target': self.model.q_1_target.state_dict(),
                'q2_target': self.model.q_2_target.state_dict(),
                'model': self.model.model.state_dict(),
            }
        }
        self.notify_epoch_handlers(full_model_state)
        returns = np.zeros((self.eval_epochs, 100), dtype=np.int8)
        all_obs = np.zeros((self.eval_epochs, 100, 32, 32, 3))
        with torch.no_grad():
            for i in range(self.eval_epochs):
                last_obs, eval_latents = self.init_latent(self.eval_env)
                for j in range(100):
                    action, _, _ = self.model.q_1.sample_max_action(eval_latents.cuda())
                    action = action.detach().cpu().numpy()
                    eval_obs, r, d, info = self.eval_env.step(action)
                    returns[i, j] = r
                    if d:
                        eval_obs, eval_latents = self.init_latent(self.eval_env, write=False)
                    else:
                        _obs = self.memory.resize(last_obs)
                        _obs = self.memory.torch_transform(_obs[np.newaxis, :]).unsqueeze(0)
                        action = self.memory.torch_transform(action[np.newaxis, :], action_expand=True)
                        eval_latents = self.model.model.update_latent(_obs, action, eval_latents.cuda())
                    last_obs = eval_obs
                    all_obs[i, j] = eval_obs
        return np.mean(np.sum(returns, -1)), np.var(np.sum(returns, -1)), all_obs
    
    def pretrain(self, batch_size, epochs, img=False):
        for epoch in tqdm(range(epochs)):
            for i in tqdm(range(len(self.memory)//batch_size)):
                batch = self.compose_batch(batch_size)
                self.model.update_model(batch, pretrain_img=img)


class SACTrainer(RLTrainer):

    def __init__(self, config, model, env, memory, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.memory = memory
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.done = False
        self.obs = self.prepend_zero(self.env.reset())
        self.eval_epochs = config.eval_epochs
        self.eval_every = 2500

        self.exploration_steps = config.exploration_steps
    
    def update(self, batch):
        return self.model.update(batch)

    def eval(self):
        full_model_state = {
            'model_state': {
                'q1': self.model.q_1.state_dict(),
                'q2': self.model.q_2.state_dict(),
                'q1_target': self.model.q_1_target.state_dict(),
                'q2_target': self.model.q_2_target.state_dict(),
            }
        }
        all_obs = np.zeros((self.eval_epochs, 100, 32, 32, 3))
        self.notify_epoch_handlers(full_model_state)
        returns = np.zeros((self.eval_epochs, 100), dtype=np.int8)
        with torch.no_grad():
            for i in range(self.eval_epochs):
                obs = self.eval_env.reset()
                obs = self.prepend_zero(obs)
                for j in range(100):
                    all_obs[i, j] = obs[3]
                    action, _, _ = self.model.q_1.sample_max_action(self.full_transform(obs))
                    action = action.detach().cpu().numpy()
                    new_obs, r, d, info = self.eval_env.step(action)
                    obs[:3] = obs[1:]
                    obs[3] = new_obs
                    returns[i, j] = r
                    if d:
                        obs = self.prepend_zero(self.eval_env.reset())
        return np.mean(np.sum(returns, -1)), np.var(np.sum(returns, -1)), all_obs

    def explore(self, initial = False):
        with torch.no_grad():
            steps = self.exploration_steps if not initial else 1000
            obs = []
            actions = []
            r_s = []
            d_s = []
            i = 0
            while i < steps:
                i += 1
                eval_obs = self.full_transform(self.obs)
                action, _, _ = self.model.q_1.sample_action(eval_obs)
                action = action.detach().cpu().numpy()
                new_obs, r, d, _ = self.env.step(action)
                obs.append(self.obs[:, 3])
                actions.append(action)
                r_s.append(r)
                d_s.append(d)
                self.obs[:3] = self.obs[1:]
                self.obs[3] = new_obs
                if d:
                    self.obs = self.prepend_zero(self.env.reset())
        obs = np.stack(obs, 0)
        actions = np.stack(actions, 0).squeeze(-1)
        r_s = np.stack(r_s, 0)
        d_s = 0. + np.stack(d_s, 0)
        self.memory.put(obs, actions, r_s, d_s)

    def prepend_zero(self, obs):
        obs = obs[np.newaxis, :, :, :]
        zeros = np.zeros_like(obs)
        return np.concatenate((zeros, zeros, zeros, obs))

    def full_transform(self, x):
        x = np.concatenate([self.memory.resize(_x)[np.newaxis, ...] for _x in x])
        x = self.memory.torch_transform(x).unsqueeze(0)
        return x
