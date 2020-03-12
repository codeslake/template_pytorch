import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import time
import getopt
import math
import numpy
import os
import PIL.Image
import sys
import collections
import numpy as np
from data_loader.data_loader import Data_Loader
import matplotlib.pyplot as plt
import gc

from models import create_model
from utils import *
from loss import *
from torch_utils import *
from ckpt_manager import CKPT_Manager

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Trainer():
    def __init__(self, config):
        self.config = config
        self.summary = SummaryWriter(config.LOG_DIR.log_scalar_train_itr)

        ## model
        print(toGreen('Loading Model...'))
        self.model = create_model(config)
        self.model.print()

        ## inputs
        print(toGreen('Initializing Input...'))
        print(toRed('\tinput type: {}'.format(self.config.type)))
        print(toRed('\tis_patch: {}'.format(self.config.is_patch)))
        inputs = {'inp':None, 'gt':None}
        self.inputs = collections.OrderedDict(sorted(inputs.items(), key=lambda t:t[0]))

        ## checkpoint manager
        self.ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num)
        if config.load_pretrained:
            print(toGreen('Loading pretrained Model...'))
            load_result = self.model.get_network().load_state_dict(torch.load(os.path.join('./ckpt', config.pre_ckpt_name)))
            lr = config.lr_init * (config.decay_rate ** (config.epoch_start // config.decay_every))
            print(toRed('\tlearning rate: {}'.format(lr)))
            print(toRed('\tload result: {}'.format(load_result)))
            self.model.set_optim(lr)

        ## data loader
        print(toGreen('Loading Data Loader...'))
        self.data_loader_train = Data_Loader(config, is_train = True, name = 'train', thread_num = config.thread_num)
        self.data_loader_test = Data_Loader(config, is_train = False, name = 'test', thread_num = config.thread_num)
        self.data_loader_train.init_data_loader(self.inputs)
        self.data_loader_test.init_data_loader(self.inputs)

        ## training vars
        self.max_epoch = 10000
        self.epoch_range = np.arange(config.epoch_start, self.max_epoch)
        self.itr_global = 0
        self.itr = 0
        self.err_epoch_train = 0
        self.err_epoch_test = 0

    def train(self):
        print(toYellow('======== TRAINING START ========='))
        for epoch in self.epoch_range:
            ## TRAIN ##
            self.itr = 0
            self.err_epoch_train = 0
            self.model.train()
            while True:
                if self.iteration(self.data_loader_train, epoch): break
            err_epoch_train = self.err_epoch_train / self.itr

            ## TEST ##
            self.itr = 0
            self.err_epoch_test = 0
            self.model.eval()
            while True:
                with torch.no_grad():
                    if self.iteration(self.data_loader_test, epoch, is_train = False): break
            err_epoch_test = self.err_epoch_test / self.itr

            ## LOG
            if epoch % self.config.write_ckpt_every_epoch == 0:
                self.ckpt_manager.save_ckpt(self.model.get_network(), epoch + 1, score = err_epoch_train)
                remove_file_end_with(self.config.LOG_DIR.sample, '*.png')

            self.summary.add_scalar('loss/epoch_train', err_epoch_train, epoch)
            self.summary.add_scalar('loss/epoch_test', err_epoch_test, epoch)

    def iteration(self, data_loader, epoch, is_train = True):
        lr = None
        itr_time = time.time()

        inputs, is_end = data_loader.get_feed()
        if is_end: return is_end

        errs, outs = self.model.get_results(inputs)

        if is_train:
            lr = self.model.update(epoch)

            if self.itr % config.write_log_every_itr == 0:
                try:
                    self.summary.add_scalar('loss/itr', errs['total'].item(), self.itr_global)
                    vutils.save_image(LAB2RGB_cv(inputs['input'].detach().cpu().numpy(), self.config.type), '{}/{}_{}_1_input.png'.format(self.config.LOG_DIR.sample, epoch, self.itr), nrow=3, padding = 0, normalize = False)
                    i = 2
                    for key, val in outs.items():
                        if val is not None:
                            vutils.save_image(LAB2RGB_cv(val.detach().cpu().numpy(), self.config.type), '{}/{}_{}_{}_out_{}.png'.format(self.config.LOG_DIR.sample, epoch, self.itr, i, key), nrow=3, padding = 0, normalize = False)
                        i += 1

                    vutils.save_image(LAB2RGB_cv(inputs['gt'].detach().cpu().numpy(), self.config.type), '{}/{}_{}_{}_gt.png'.format(self.config.LOG_DIR.sample, epoch, self.itr, i), nrow=3, padding = 0, normalize = False)
                except Exception as ex:
                    print('saving error: ', ex)

            print_logs('TRAIN', self.config.mode, epoch, itr_time, self.itr, data_loader.num_itr, errs = errs, lr = lr)
            self.err_epoch_train += errs['total'].item()
            self.itr += 1
            self.itr_global += 1
        else:
            print_logs('TEST', self.config.mode, epoch, itr_time, self.itr, data_loader.num_itr, errs = errs)
            self.err_epoch_test += errs['total'].item()
            self.itr += 1

        gc.collect()
        return is_end

##########################################################

if __name__ == '__main__':
    print(toGreen('Laoding Config...'))

    import argparse
    from config import get_config, log_config, print_config

    project = 'Project'
    mode = 'project'

    config = get_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = mode, help = 'model name')
    parser.add_argument('-t', '--is_train', action = 'store_true', default = config.is_train, help = 'whether to train')
    parser.add_argument('-dl', '--delete_log', action = 'store_true', default = config.delete_log, help = 'whether to delete log')
    parser.add_argument('-lr', '--lr_init', type = float, default = config.lr_init, help = 'leraning rate')
    parser.add_argument('-th', '--thread_num', type = int, default = config.thread_num, help = 'number of thread')
    parser.add_argument('-b', '--batch_size', type = int, default = config.batch_size, help = 'number of batch')

    parser.add_argument('-md', '--model', type = str, default = config.model, help = 'model name')

    parser.add_argument('-lp', '--load_pretrained', action = 'store_true', default = config.load_pretrained, help = 'wheter to load pretrained checkpoint')
    parser.add_argument('-pckn', '--pre_ckpt_name', type = str, default = config.pre_ckpt_name, help = 'name of pretrained checkpoint to load')

    parser.add_argument('-es', '--epoch_start', type = int, default = config.epoch_start, help = 'whether to apply semantic replacement')

    args = parser.parse_args()
    config = get_config(project, args.mode)

    config.is_train = args.is_train
    config.delete_log = args.delete_log
    config.lr_init = args.lr_init
    config.thread_num = args.thread_num
    config.batch_size = args.batch_size

    config.model = args.model

    config.load_pretrained = args.load_pretrained
    config.load_pretrained_partial = args.load_pretrained_partial

    config.epoch_start = args.epoch_start

    handle_directory(config, args.delete_log)
    print_config(config)
    log_config(config.LOG_DIR.config, config)

    if is_train:
        trainer = Trainer(config)
        trainer.train()
    else:
        return
