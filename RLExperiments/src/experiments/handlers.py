from abc import ABC, abstractmethod
import os, shutil

from src.util.tf_logger import Logger


class Handler(ABC):
    @abstractmethod
    def run(self, model, data):
        pass

    @abstractmethod
    def reset(self):
        pass


class PrintHandler(Handler):
    def run(self, model, data):
        print(data)

    def reset(self):
        pass


class TensorboardHandler(Handler):

    def __init__(self, logdir='../logs', reset_logdir=True):
        self.logger = Logger(logdir)
        if reset_logdir:
            for the_file in os.listdir(logdir):
                file_path = os.path.join(logdir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        self.step = 0

    def run(self, model, data):
        for tag, value in data['tf_logging'].items():
            self.logger.scalar_summary(tag, value, self.step)
        if 'step' in data:
            self.step += data['step']
        self.step += 1

    def reset(self):
        self.step = 0
        self.logger = Logger('../logs')


class VisdomHandler(Handler):
    def run(self, model, data):
        pass

    def reset(self):
        pass


class SaveHandler(Handler):
    def run(self, model, data):
        pass

    def reset(self):
        pass


class RenderHandler(Handler):
    def run(self, model, data):
        kwargs['env'].render()
    
    def reset(self):
        pass
