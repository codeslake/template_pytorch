import torch
import collections

from utils import *
from models.utils import *
from data_loader.data_loader import Data_Loader

class baseModel():
    config = None

    network = None
    visuals = collections.OrderedDict()

    def _set_optim(self, lr = None):
        print(toGreen('Building Optim...'))
        self._set_loss()
        lr = self.config.lr_init if lr is None else lr
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(self.config.beta1, 0.999))

    def _set_dataloader(self, inputs):
        print(toGreen('Loading Data Loader...'))
        self.data_loader_train = Data_Loader(self.config, is_train = True, name = 'train', thread_num = self.config.thread_num)
        self.data_loader_test = Data_Loader(self.config, is_train = False, name = 'test', thread_num = self.config.thread_num)
        self.data_loader_train.init_data_loader(inputs)
        self.data_loader_test.init_data_loader(inputs)

    def _updatr(self, epoch, errs):
        self.optimizer.zero_grad()
        errs['total'].backward()
        self.optimizer.step()
        lr = adjust_learning_rate(self.optimizer, epoch, self.config.decay_rate, self.config.decay_every, self.config.lr_init)

        return lr

    def get_inputs(self):
        return self.inputs

    def get_network(self):
        return self.network

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def print(self):
        print(self.network)
