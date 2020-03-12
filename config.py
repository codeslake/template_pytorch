from easydict import EasyDict as edict
import json
import os
import collections

def get_config(project = '', mode = ''):
    ## GLOBAL
    config = edict()
    config.project = project
    config.mode = mode
    config.is_train = True
    config.thread_num = 1

    ##################################### TRAIN #####################################
    config.model = 'input_add'

    config.batch_size = 1
    config.height = 256
    config.width = 256
    config.load_pretrained = False
    config.pre_ckpt_name = None

    # traine
    config.epoch_start = 0
    config.n_epoch = 10000

    # learning rate
    config.lr_init = 5e-5
    config.decay_rate = 0.7
    config.decay_every = 10
    config.grad_norm_clip_val = 1.0

    # data
    config.is_color = True
    config.is_augment = True
    config.is_reverse = True

    # adam
    config.beta1 = 0.9

    # data dir
    config.data_path = ''
    config.input_path = 'input' #os.path.join(config.data_path, 'input')
    config.gt_path = 'gt' #os.path.join(config.data_path, 'gt')

    # dataloader
    config.skip_length = [0]

    # logs
    config.max_ckpt_num = 10
    config.write_ckpt_every_epoch = 1
    config.refresh_image_log_every_itr = 10000
    config.refresh_image_log_every_epoch = 2
    config.write_log_every_itr = 20
    config.write_ckpt_every_itr = 1000

    # log dirs
    config.LOG_DIR = edict()
    offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(offset, config.project)
    offset = os.path.join(offset, '{}'.format(mode))
    config.LOG_DIR.ckpt = os.path.join(offset, 'checkpoint', 'train', 'epoch')
    config.LOG_DIR.ckpt_itr = os.path.join(offset, 'checkpoint', 'train', 'itr')
    config.LOG_DIR.log_scalar_train_epoch = os.path.join(offset, 'log', 'train', 'scalar', 'train', 'epoch')
    config.LOG_DIR.log_scalar_train_itr = os.path.join(offset, 'log', 'train', 'scalar', 'train', 'itr')
    config.LOG_DIR.log_scalar_valid = os.path.join(offset, 'log', 'train', 'scalar', 'valid')
    config.LOG_DIR.log_image = os.path.join(offset, 'log', 'train', 'image', 'train')
    config.LOG_DIR.sample = os.path.join(offset, 'sample', 'train')
    config.LOG_DIR.config = os.path.join(offset, 'config')

    ################################## VALIDATION ###################################
    # data path
    offset = ''
    config.VAL = edict()
    config.VAL.data_path = ''
    config.VAL.input_path = 'input' #os.path.join(config.VAL.data_path, 'input')
    config.VAL.gt_path = 'gt' #os.path.join(config.VAL.data_path, 'gt')

    ##################################### EVAL ######################################
    config.EVAL = edict()
    config.EVAL.eval_mode = 'eval'
    config.EVAL.load_ckpt_by_score = True
    config.EVAL.ckpt_name = None

    # data dir
    offset = ''
    config.EVAL.input_path = os.path.join(offset, 'input')
    config.EVAL.gt_path = os.path.join(offset, 'gt')
    config.EVAL.LOG_DIR = edict()

    # log dir
    offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(offset, config.project)
    offset = os.path.join(offset, '{}'.format(config.mode))
    config.EVAL.LOG_DIR.save = os.path.join(offset, 'result')

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')


def print_config(cfg):
    print(json.dumps(cfg, indent=4))

