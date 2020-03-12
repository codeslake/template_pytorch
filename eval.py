import torch 
import torchvision.utils as vutils
import torch.nn.functional as F

import datetime
import time
import os
import collections
import random
import cv2
from threading import Thread
from shutil import rmtree
import numpy as np
from pathlib import Path

from model.model import create_model
from utils import *
from torch_utils import *

from ckpt_manager import CKPT_Manager

import collections

def evaluate(config, mode):
    print(toGreen('Loading checkpoint manager...'))
    print(config.LOG_DIR.ckpt, mode)
    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, mode, 10)

    date = datetime.datetime.now().strftime('%Y.%m.%d.(%H%M)')
    print(toYellow('======== EVALUATION START ========='))
    ##################################################
    ## DEFINE MODEL
    print(toGreen('Initializing model'))
    model = create_model(config)
    model.eval()
    model.print()

    inputs = {'inp':None, 'ref':None, 'inp_hist':None, 'ref_hist':None, 'seg_inp':None, 'seg_ref':None}
    inputs = collections.OrderedDict(sorted(inputs.items(), key=lambda t:t[0]))

    ## INITIALIZING VARIABLE
    print(toGreen('Initializing variables'))
    result, ckpt_name = ckpt_manager.load_ckpt(model.get_network(), by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name)
    print(result)
    save_path_root = os.path.join(config.EVAL.LOG_DIR.save, config.EVAL.eval_mode, ckpt_name, date)
    exists_or_mkdir(save_path_root)
    torch.save(model.get_network().state_dict(), os.path.join(save_path_root, ckpt_name + '.pytorch'))
    
    ##################################################
    inp_folder_path_list, _, _ = load_file_list(config.EVAL.inp_path)
    ref_folder_path_list, _, _ = load_file_list(config.EVAL.ref_path)
    inp_segmap_folder_path_list, _, _ = load_file_list(config.EVAL.inp_segmap_path)
    ref_segmap_folder_path_list, _, _ = load_file_list(config.EVAL.ref_segmap_path)

    print(toGreen('Starting Color Trasfer'))
    itr = 0
    for folder_idx in np.arange(len(inp_folder_path_list)):
        inp_folder_path = inp_folder_path_list[folder_idx]
        ref_video_path = ref_folder_path_list[folder_idx]
        inp_segmap_folder_path = inp_segmap_folder_path_list[folder_idx]
        ref_segmap_video_path = ref_segmap_folder_path_list[folder_idx]

        _, inp_file_path_list, _ = load_file_list(inp_folder_path)
        _, ref_file_path_list, _ = load_file_list(ref_video_path)
        _, inp_segmap_file_path_list, _ = load_file_list(inp_segmap_folder_path)
        _, ref_segmap_file_path_list, _ = load_file_list(ref_segmap_video_path)
        for file_idx in np.arange(len(inp_file_path_list)):
            inp_path = inp_file_path_list[file_idx]
            ref_path = ref_file_path_list[file_idx]
            inp_segmap_path = inp_segmap_file_path_list[file_idx]
            ref_segmap_path = ref_segmap_file_path_list[file_idx]

            # inputs['inp'], inputs['inp_hist'] = torch.FloatTensor(_read_frame_cv(inp_path, config).transpose(0, 3, 1, 2)).cuda()

            inputs['inp'], inputs['inp_hist'] = _read_frame_cv(inp_path, config)
            inputs['ref'], inputs['ref_hist'] = _read_frame_cv(ref_path, config)
            inputs['seg_inp'], inputs['seg_ref'] = _read_segmap(inputs['inp'], inp_segmap_path, ref_segmap_path, config.is_rep)

            p = 30
            reppad = torch.nn.ReplicationPad2d(p)

            for key, val in inputs.items():
                inputs[key] = torch.FloatTensor(inputs[key].transpose(0, 3, 1, 2)).cuda()

            inputs['inp'] = reppad(inputs['inp'])
            inputs['ref'] = reppad(inputs['ref'])
            inputs['seg_inp'] = reppad(inputs['seg_inp'])
            inputs['seg_ref'] = reppad(inputs['seg_ref'])

            with torch.no_grad():
                outs = model.get_results(inputs)

            inputs['inp'] = inputs['inp'][:,:,p:(inputs['inp'].size(2) - p),p:(inputs['inp'].size(3) - p)]
            inputs['ref'] = inputs['ref'][:,:,p:(inputs['ref'].size(2) - p),p:(inputs['ref'].size(3) - p)]
            outs['result'] = outs['result'][:,:,p:(outs['result'].size(2) - p),p:(outs['result'].size(3) - p)]
            outs['result_idt'] = outs['result_idt'][:,:,p:(outs['result_idt'].size(2) - p),p:(outs['result_idt'].size(3) - p)]
            outs['seg_inp'] = outs['seg_inp'][:,:,p:(outs['seg_inp'].size(2) - p),p:(outs['seg_inp'].size(3) - p)]
            outs['seg_ref'] = outs['seg_ref'][:,:,p:(outs['seg_ref'].size(2) - p),p:(outs['seg_ref'].size(3) - p)]

            file_name = os.path.basename(inp_file_path_list[file_idx]).split('.')[0]
            save_path = save_path_root
            exists_or_mkdir(save_path)

            vutils.save_image(LAB2RGB_cv(inputs['inp'].detach().cpu(), config.type), '{}/{}_1_inp.png'.format(save_path, itr), nrow=3, padding = 0, normalize = False)
            vutils.save_image(LAB2RGB_cv(inputs['ref'].detach().cpu(), config.type), '{}/{}_2_ref.png'.format(save_path, itr), nrow=3, padding = 0, normalize = False)
            i = 3
            out_keys = ['result', 'result_idt', 'seg_inp', 'seg_ref']
            for key, val in outs.items():
                if key in out_keys:
                    if val is not None:
                        if 'seg' in key:
                            if config.identity is False and 'idt' in key:
                                continue
                            vutils.save_image(LAB2RGB_cv(val.detach().cpu(), 'rgb') / 8., '{}/{}_{}_out_{}.png'.format(save_path, itr, i, key), nrow=3, padding = 0, normalize = False)
                        else:
                            if config.identity is False and 'idt' in key:
                                continue
                            vutils.save_image(LAB2RGB_cv(val.detach().cpu(), config.type), '{}/{}_{}_out_{}.png'.format(save_path, itr, i, key), nrow=3, padding = 0, normalize = False)
                        i += 1

            #PSNR
            print('[{}][{}/{}'.format(ckpt_name, folder_idx+1, len(inp_folder_path_list)), '{}/{}]'.format(file_idx+1, len(inp_file_path_list)))
            itr += 1
            ##################################################

def _read_segmap(inp, inp_path, ref_path, is_rep):
    seg1 = cv2.imread(inp_path)
    seg2 = cv2.imread(ref_path)

    seg1, seg2, segnum = MakeLabelFromMap(seg1, seg2)
    seg1 = np.expand_dims(seg1, axis = 0)
    seg2 = np.expand_dims(seg2, axis = 0)

    return seg1, seg2

def _read_frame_cv(path, config):

    image_bgr = cv2.imread(path) / 255.

    if config.type == 'lab':
        image_lab, image_rgb = RGB2LAB_cv(image_bgr, config.type)
        image_lab = np.expand_dims(image_lab, axis = 0)

        if config.H_enc_method == 'l_ab':
            hist_ab = getHistogram2d_np(image_lab, config.nAB, config.hist_include_0)
            hist_l = getHistogram1d_np(image_lab, config.nL, config.hist_include_0)
            hist_lab = np.concatenate([hist_ab, hist_l], axis = 3)
        elif config.H_enc_method == 'la_lb_ab':
            hist_la = getHistogram2d_np(image_lab, config.nAB, config.hist_include_0, axis1 = 0, axis2 = 1)
            hist_lb = getHistogram2d_np(image_lab, config.nAB, config.hist_include_0, axis1 = 0, axis2 = 2)
            hist_ab = getHistogram2d_np(image_lab, config.nAB, config.hist_include_0, axis1 = 1, axis2 = 2)

            hist_lab = np.concatenate([hist_la, hist_lb, hist_ab], axis = 3)

        return image_lab, hist_lab

    elif config.type == 'rgb':
        image_rgb = RGB2HSV_shift2RGB_cv(image_bgr)
        image_rgb = np.expand_dims(image_rgb, axis = 0)

        if config.H_enc_method == 'rgb3d':
            hist_rgb = getHistogram3d_np(image_rgb, config.nRGB, config.hist_include_0)
        if config.H_enc_method == 'rgb1d':
            hist_r = getHistogram1d_np(image_rgb, 32, config.hist_include_0, axis = 0, is_tile = False)
            hist_g = getHistogram1d_np(image_rgb, 32, config.hist_include_0, axis = 1, is_tile = False)
            hist_b = getHistogram1d_np(image_rgb, 32, config.hist_include_0, axis = 2, is_tile = False)
            hist_rgb = np.concatenate([hist_r, hist_g, hist_b], axis = 3)

        return image_rgb, hist_rgb

def handle_directory(config, delete_log):
    def mkdir(dir_dict, delete_log_, delete_ckpt_ = True):
        for (key, val) in dir_dict.items():
            if 'perm' in key and delete_ckpt_ is False:
                exists_or_mkdir(val)
                continue

            if delete_log_:
                rmtree(val, ignore_errors = True)
            exists_or_mkdir(val)

    delete_log = delete_log

    if delete_log:
        delete_log = input('Are you sure to delete the logs (y/n): ')

        if len(delete_log) == 0 or delete_log[0].lower() == 'y':
            delete_log = True
        elif delete_log[0].lower() == 'n':
            delete_log = False
        else:
            print('invalid input')
            exit()

    if 'is_pretrain' in list(config.keys()) and config.is_pretrain:
        delete_ckpt = True if config.PRETRAIN.delete_log else False
        mkdir(config.PRETRAIN.LOG_DIR, delete_log, delete_ckpt)
    mkdir(config.LOG_DIR, delete_log)

if __name__ == '__main__':
    import argparse
    from config import get_config, log_config, print_config

    parser = argparse.ArgumentParser()
    parser.add_argument('-em', '--eval_mode', type=str, default = 'eval', help = 'limits of losses that controls coefficients')
    parser.add_argument('-m', '--mode', type = str, default = 'color_transfer_', help = 'model name')
    parser.add_argument('-ckpt_sc', '--load_ckpt_by_score', type=str, default = 'true', help = 'limits of losses that controls coefficients')
    parser.add_argument('-ckpt_name', '--ckpt_name', type=str, default = None, help = 'limits of losses that controls coefficients')
    parser.add_argument('-md', '--model', type = str, default ='input_end', help = 'model name')
    parser.add_argument('-it', '--is_train', type = str, default ='False', help = 'model name')
    parser.add_argument('-rm', '--rep_mode', type = str, default ='gsgt', help = 'model name')


    args = parser.parse_args()

    config = get_config(args.mode)

    config.mode = args.mode
    config.EVAL.load_ckpt_by_score = to_bool(args.load_ckpt_by_score)
    config.EVAL.ckpt_name = args.ckpt_name
    config.EVAL.eval_mode = args.eval_mode
    config.model = args.model
    config.is_train = to_bool(args.is_train)
    config.rep_mode = args.rep_mode

    print(toWhite('Creating log directories...'))
    handle_directory(config.EVAL, False)

    print(toYellow('\n[TESTING {}]\n'.format(config.mode)))
    evaluate(config, config.mode)
