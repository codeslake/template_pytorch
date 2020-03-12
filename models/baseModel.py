import torch
import torch.nn as nn

from utils import *
from torch_utils import *

class baseModel():
    network = None
    config = None
    errs = collections.OrderedDict()
    outs = collections.OrderedDict()

    def set_optim(self, lr = None):
        print(toGreen('Building Optim...'))
        lr = self.config.lr_init if lr is None else lr
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(self.config.beta1, 0.999))

    def update(self, epoch):
        self.optimizer.zero_grad()
        self.errs['total'].backward()
        self.optimizer.step()
        lr = adjust_learning_rate(self.optimizer, epoch, self.config.decay_rate, self.config.decay_every, self.config.lr_init)
        
        return lr

    def get_network(self):
        return self.network

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def print(self):
        print(self.network)
