import torch
import numpy as np

from torch_runner.train.base import AbstractTrainer

from util.visdom import VisdomLogger

class MONetTrainer(AbstractTrainer):

    def train_step(self, data, **kwargs):
        if 'threshold' in kwargs.keys():
            loss, data_dict = self.model(data, 0.2)
        else:
            loss, data_dict = self.model(data)
        self.optimizer.zero_grad()
        torch.mean(loss).backward()
        if self.clip_gradient:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient_value)
        self.optimizer.step()
        # if self.num_steps % 150 == 0 and kwargs['visdom']:
        #     self.visdom_logger.log_visdom(data, data_dict['masks'], data_dict['reconstruction'])
        # self.num_steps += 1
        return data_dict

    def check_ready(self):
        if self.train_dataloader is None:
            return False
        if self.optimizer is None:
            return False
        if self.model is None:
            return False
        # self.visdom_logger = VisdomLogger(8456, 'quick_try_4')
        self.num_steps = 0
        return True

    def compile_epoch_info_dict(self, data_dict, epoch, **kwargs):
        # self.model.beta = 1 / (1 + np.exp(-(epoch/4 - 10)))
        # self.model.gamma = 1 / (1 + np.exp(-(epoch/4 - 10)))
        return {'model_state': self.model.state_dict()}


class GECOTrainer(AbstractTrainer):

    def ready(self):
        self.beta = torch.tensor(0.5).cuda()
        self.err_ema = 1.
        return True

    def train_step(self, data):
        _, data_dict = self.model(data)
        err = -1. * data_dict['p_x'].mean()
        kl_m = data_dict['kl_m_loss'].mean()
        kl_r = data_dict['kl_r_loss'].mean()
        loss = err + self.beta * (kl_m + kl_r)
        
        err_new = err.detach()
        kl_new = (kl_m + kl_r).detach()
        self.err_ema = self.get_ema(err_new, self.err_ema, 0.99)
        self.beta = self.geco_beta_update(
            self.beta, 
            self.err_ema, 
            1., 
            1e-5)

        self.optimizer.zero_grad()
        torch.mean(loss).backward()
        if self.clip_gradient:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient_value)
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

    def compile_epoch_info_dict(self, data_dict, epoch):
        self.model.beta = 1 / (1 + np.exp(-(epoch - 10)))
        return {'model_state': self.model.state_dict()}

    def get_ema(self, new, old, alpha):
        if old is None:
            return new
        return (1.0 - alpha) * new + alpha * old

    def geco_beta_update(self, beta, error_ema, goal, step_size,
                         min_clamp=1e-10):
        constraint = (goal - error_ema).detach()
        beta = beta * torch.exp(step_size * constraint)
        # Clamp beta to be larger than minimum value
        if min_clamp is not None:
            beta = torch.max(beta, torch.tensor(min_clamp).cuda())
        # Detach again just to be safe
        return beta.detach()
