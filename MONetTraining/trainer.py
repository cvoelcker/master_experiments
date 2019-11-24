import torch

from torch_runner.train.base import AbstractTrainer

class MONetTrainer(AbstractTrainer):

    def train_step(self, data):
        loss, data_dict = self.model(data)
        self.optimizer.zero_grad()
        torch.mean(loss).backward()
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
