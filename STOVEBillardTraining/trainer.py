import torch

from torch_runner.train.base import AbstractTrainer

class MONetTrainer(AbstractTrainer):

    def train_step(self, data):
        # torch.autograd.set_detect_anomaly(True)
        loss, data_dict, r = self.model(data['X'], 0, data['action'].cuda().float())
        self.optimizer.zero_grad()
        torch.mean(-1. * loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
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

    def compile_epoch_info_dict(self, data_dict):
        return {'model_state': self.model.state_dict()}


class MONetTester(AbstractTrainer):

    def train_step(self, data):
        with torch.no_grad():
        # torch.autograd.set_detect_anomaly(True)
            loss, data_dict, r = self.model(data['X'], 0, data['action'].cuda().float())
            z_full, imgs, r = self.model.rollout(
                    data_dict['z_s'][:, -1].cuda().float(),
                    num = 7,
                    actions=data['action'].cuda().float(),
                    return_imgs=True)
            return {'z': z_full, 'r': r, 'imgs': imgs}

    def check_ready(self):
        if self.train_dataloader is None:
            return False
        if self.optimizer is None:
            return False
        if self.model is None:
            return False
        return True

    def compile_epoch_info_dict(self, data_dict):
        return {'model_state': self.model.state_dict()}
