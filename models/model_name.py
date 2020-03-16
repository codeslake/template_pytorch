import torch
import torch.nn as nn
import numpy as np
import collections

from utils import *
from model.utils import *
from models.baseModel import baseModel
from models.archs.network_name import Network as N
from models.archs.network_name import Network2 as N2

class Model(baseModel):
    def __init__(self, config):
        self.config = config

        inputs = {'input':None, 'gt':None}
        inputs = collections.OrderedDict(sorted(inputs.items(), key=lambda t:t[0]))
        
        self.network = Network()

        self.network.init()
        self.network = torch.nn.DataParallel(self.network).cuda()

        self._set_optim()
        self._set_dataloader(inputs)

    def iteration(self, epoch, is_train):
        data_loader = self.data_loader_train if is_train else self.data_loader_test

        inputs, is_end = data_loader.get_feed()
        if is_end: return is_end

        errs, outs = self._get_results(inputs)

        if is_train:
            lr = self._update(epoch)

        self.visuals['inputs'] = inputs
        self.visuals['errs'] = errs
        self.visuals['outs'] = outs
        self.visuals['lr'] = lr
        self.visuals['num_itr'] = data_loader.num_itr

        return is_end

    def _set_loss(self, lr = None):
        print(toGreen('Building Loss...'))
        self.MSE = torch.nn.MSELoss().cuda()
        self.MAE = torch.nn.L1Loss().cuda()

    def _get_results(self, inputs):
        ## network output
        outs = collections.OrderedDict()
        outs['result'] = self.network(inputs['input'], I_prev_debl)

        ## loss
        if self.config.is_train:
            # self.image_loss = self.MAE if self.config.loss == 'MAE' else self.MSE
            
            errs = collections.OrderedDict()
            errs['image'] = self.MSE(outs['result'], inputs['gt'])
            errs['total'] = errs['image']
                             
            return errs, outs
        else:
            return outs

# Unet based
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.N = N()
        self.N2 = N2()

    def init(self):
        self.N1.load_state_dict(torch.load('./ckpt/network-default.pytorch'))
        self.N2.apply(weights_init)

    def forward(self, inp):
        N_out = self.N(inp)
        N2_out = self.N2(inp)

        return N_out + N2_out

