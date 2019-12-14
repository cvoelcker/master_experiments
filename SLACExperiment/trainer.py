import numpy as np
import torch
import copy

from tqdm import tqdm

from torch_runner.train.base import AbstractTrainer

class RLTrainer(AbstractTrainer):

    def __init__(self, config, model, env, memory, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.memory = memory
        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.done = False
        self.obs, self.latents = self.init_latent(self.env)
        self.eval_epochs = 100
        self.eval_every = 5000

        self.exploration_steps = config.exploration_steps
    
    def check_ready(self):
        return True

    def train_step(self):
        pass

    def train(self, total_steps, batch_size, debug=False):
        # self.explore(initial=True)
        pbar = tqdm(total=total_steps)
        i = 0
        steps = 0
        while steps < total_steps:
            if (steps % self.eval_every == 0) and not debug:
                eval_mean, eval_var = self.eval()
                print(f'{eval_mean}, {eval_var}')
            batch = self.compose_batch(batch_size)
            losses = self.update(batch)
            losses['rolling_reward'] = torch.tensor(self.rolling_reward()).float()
            self.notify_step_handlers(losses)
            self.explore()
            pbar.update(1)
            # print(f'Seen {self.memory.total_samples} steps')
            steps += 1
    
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
                action, _, _ = self.model.policy.sample(self.latents.cuda())
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
    
    def update(self, batch):
        return self.model.update(batch)

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

    def eval(self):
        full_model_state = {
            'model_state': {
                'policy': self.model.policy.state_dict(),
                'q1': self.model.q_1.state_dict(),
                'q2': self.model.q_2.state_dict(),
                'q1_target': self.model.q_1_target.state_dict(),
                'q2_target': self.model.q_2_target.state_dict(),
                'model': self.model.model.state_dict(),
            }
        }
        self.notify_epoch_handlers(full_model_state)
        returns = np.zeros((self.eval_epochs, 100), dtype=np.int8)
        with torch.no_grad():
            for i in range(self.eval_epochs):
                last_obs, eval_latents = self.init_latent(self.eval_env)
                for j in range(100):
                    action, _, _ = self.model.policy.sample(eval_latents.cuda())
                    action = action.detach().cpu().numpy()
                    eval_obs, r, d, info = self.env.step(action)
                    returns[i, j] = r
                    if d:
                        eval_obs, eval_latents = self.init_latent(self.eval_env, write=False)
                    else:
                        _obs = self.memory.resize(last_obs)
                        _obs = self.memory.torch_transform(_obs[np.newaxis, :]).unsqueeze(0)
                        action = self.memory.torch_transform(action[np.newaxis, :], action_expand=True)
                        eval_latents = self.model.model.update_latent(_obs, action, eval_latents.cuda())
                    last_obs = eval_obs
        return np.mean(np.sum(returns, -1)), np.var(np.sum(returns, -1))

    def rolling_reward(self):
        fill = len(self.memory)
        rewards = self.memory.r_buffer[fill-1000:fill]
        return np.sum(rewards)


class MONetTrainer(AbstractTrainer):

    def train_step(self, data, **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        loss, data_dict, r = self.model(data['X'], actions=data['action'].cuda().float(), pretrain=kwargs['pretrain'])
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
            self.model.img_model.beta = 1 / (1 + np.sigmoid(epoch))
        return {'model_state': self.model.state_dict(),}
                #'imgs': (data_dict['imgs'] * 255).type_as(torch.ByteTensor())}


class MONetTester(AbstractTrainer):

    def train_step(self, data):
        with torch.no_grad():
        # torch.autograd.set_detect_anomaly(True)
            print(data['X'].shape)
            loss, data_dict, r = self.model(data['X'], actions=data['action'].cuda().float())
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