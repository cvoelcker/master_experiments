import numpy as np
import torch

from torch_runner.train.base import AbstractTrainer
from util.visdom import VisdomLogger

class MONetTrainer(AbstractTrainer):

    def train_step(self, data, **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        loss, data_dict, r = self.model(data['X'], actions=data['action'].cuda().float(), rewards=data['reward'].cuda().float().squeeze(-1) ,mask=data['done'].cuda().float().squeeze(-1), pretrain=kwargs['pretrain'])
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
        # if self.num_steps % 150 == 0 and kwargs['visdom']:
        #     if kwargs['pretrain']:
        #         self.visdom_logger.log_visdom(data['X'].flatten(end_dim=1), data_dict['masks'], data_dict['reconstruction'], 6)
        #     else:
        #         self.visdom_logger.log_visdom(data['X'][0], data_dict['masks'][0], data_dict['reconstruction'][0], 6)
        self.num_steps += 1
        
        return data_dict

    def check_ready(self):
        if self.train_dataloader is None:
            return False
        if self.optimizer is None:
            return False
        if self.model is None:
            return False
        # self.visdom_logger = VisdomLogger(8456, 'main_stove')
        self.num_steps = 0
        return True

    def append_epoch_info_dict(self, epoch_info_dict, data_dict):
        return data_dict

    def compile_epoch_info_dict(self, data_dict, epoch, **kwargs):
        if 'pretrain' in kwargs and kwargs['pretrain']:
            self.model.img_model.beta = 1 / (1 + np.exp(-epoch))
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
            imgs_recon = (data_dict['reconstruction'] * 255.).cpu().detach().type_as(torch.ByteTensor())
            imgs_roll = (imgs * 255).cpu().detach().type_as(torch.ByteTensor())
            imgs = torch.cat([imgs_recon, imgs_roll], 1)
            return {'z': z_full, 'r': r, 'imgs': imgs}

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


class MONetInferenceTester(AbstractTrainer):

    def train_step(self, data):
        with torch.no_grad():
        # torch.autograd.set_detect_anomaly(True)
            # loss, data_dict, r = self.model(data['X'], actions=data['action'].cuda().float())
            loss, data_dict, r = self.model(data['X'], actions=data['action'].cuda().float(), rewards=data['reward'].cuda().float().squeeze(-1) ,mask=data['done'].cuda().float().squeeze(-1))
            pic = data['X'].cpu().detach()
            new_pic = data_dict['reconstruction'].cpu().detach()
            return {
                'z': data_dict['z_s'].cpu().detach(), 
                'imgs': (data['X'] * 255).type_as(torch.ByteTensor()).cpu().detach(),
                'imgs_inferred': (data_dict['reconstruction'] * 255).type_as(torch.ByteTensor()).cpu().detach(),
                'mse': torch.sum((pic - new_pic) ** 2, (-3, -2, -1))}

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
