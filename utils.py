import numpy as np
import cv2
import math
import operator
import collections
import os
import fnmatch
import termcolor
import time
import string
from shutil import rmtree
from skimage import color
import torch
import random

def refine_image(img, val = 16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val]

def get_file_path(path, regex):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for i in np.arange(len(regex)):
            for filename in fnmatch.filter(filenames, regex[i]):
                file_path.append(os.path.join(root, filename))

    return file_path

def remove_file_end_with(path, regex):
    file_paths = get_file_path(path, [regex])

    for i in np.arange(len(file_paths)):
        os.remove(file_paths[i])

def norm_image(image, axis = (1, 2, 3)):
    image = image - np.amin(image, axis = axis, keepdims=True)
    image = image / np.amax(image, axis = axis, keepdims=True)
    return image

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])

def print_logs(train_mode, mode, epoch, time_s, iter = '', iter_total = '', errs = '', coefs = '', lr = None):
    err_str = ''
    if errs != '':
        for key, val in errs.items():
            if key == 'total':
                err_str = '{}: '.format(key) + toRed('{:1.2e}'.format(val)) + err_str
            else:
                if coefs == '':
                    err_str += ', {}: '.format(key) + toBlue('{:1.2e}'.format(val))
                elif key in list(coefs.keys()):
                    err_str += ', {}: '.format(key) + toBlue('{:1.2e}(*{:1.1e})'.format(val, coefs[key]))

        err_str = toWhite('*LOSS->') + '[' + err_str + ']'

    iter_str = ''
    if iter != '':
        iter_str = ' ({}/{})'.format(toCyan('{:04}'.format(iter + 1)), toCyan('{:04}'.format(iter_total)))

    lr_str = ''
    if lr is not None:
        lr_str = ' lr: {}'.format(toGrey('{:1.2e}'.format(lr)))

    print('[{}][{}]{}{}{}{}\n{}\n'.format(
            toWhite(train_mode),
            toYellow(mode),
            toWhite(' {} '.format('EP')) + toCyan('{}'.format(epoch + 1)),
            iter_str,
            lr_str,
            toGreen(' {:5.2f}s'.format(time.time() - time_s)),
            err_str,
            )
        )

def get_dict_with_list(list_key, list_val, default_val = None):

    is_multi_dim = False
    if type(list_val) == list:
        for val in list_val:
            if type(val) == list:
                is_multi_dim = True
                break

    new_dict = collections.OrderedDict()
    for i in np.arange(len(list_key)):
        # epoch range
        if is_multi_dim:
            if len(list_key) == len(list_val):
                list_temp = list_val[i]
            else:
                list_temp = list_val[0]
            if list_temp[1] == -1:
                new_dict[list_key[i]] = [list_temp[0], default_val]
            else:
                new_dict[list_key[i]] = list_temp
        else:
            if type(list_val) is list and len(list_val) == len(list_key):
                new_dict[list_key[i]] = list_val[i]
            else:
                new_dict[list_key[i]] = list_val

    return new_dict

def dict_operations(dict1, op, operand2):
    ops = {'+': operator.add,
           '-': operator.sub,
           '*': operator.mul,
           '/': operator.truediv
           }
    if op != '=':
        if 'dict' in str(type(dict1)).lower() and type(dict1) == type(operand2):
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2[key]) for key in dict1.keys()]))
        elif type(operand2) == list:
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2[count]) for count, key in enumerate(dict1.keys())]))
        elif type(operand2) == int or type(operand2) == float:
            return collections.OrderedDict(zip(list(dict1.keys()), [ops[op](dict1[key], operand2) for key in dict1.keys()]))
    else:
        new_dict = collections.OrderedDict()
        for key in dict1.keys():
            new_dict[dict1[key]] = operand2[key]

        return new_dict

def string_to_array(text):
    x = [word.strip(string.punctuation) for word in text.split()]
    return x

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

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
        delete_log = input('Are you sure to delete the logs (y/n):\n') # python3

        if len(delete_log) == 0 or delete_log[0].lower() == 'y':
            delete_log = True
        elif delete_log[0].lower() == 'n':
            delete_log = False
        else:
            print('invalid input')
            exit()

    mkdir(config.LOG_DIR, delete_log)

def load_file_list(root_path, child_path = None):
    folder_paths = []
    filenames_pure = []
    filenames_structured = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        if len(dirnames) != 0:
            if dirnames[0][0] == '@':
                del(dirnames[0])

        if len(dirnames) == 0:
            if root[0] == '.':
                continue
            if child_path is not None and child_path not in root: 
                continue
            folder_paths.append(root)
            filenames_pure = []
            for i in np.arange(len(filenames)):
                if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                    filenames_pure.append(os.path.join(root, filenames[i]))
            filenames_structured.append(np.array(sorted(filenames_pure)))
            num_files += len(filenames_pure)

    folder_paths = np.array(folder_paths)
    filenames_structured = np.array(filenames_structured)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    filenames_structured = filenames_structured[sort_idx]

    return folder_paths, np.squeeze(filenames_structured), np.squeeze(num_files)

def RGB2HSV_shift2LAB(I, start_time):
    this_time=time.time()
    elapsed_time=this_time-start_time
    shift = ((elapsed_time*1000)%1000)*(1.0)/1000 # above 1
    shift2 = ((elapsed_time*10000)%10000)*(1.2)/10000 # above 1 # Added at 'colorhistogram_noGAN_lr00005_lab_hueshift_histenc_histloss_satadded' 
    if shift2 < 0.3:
        shift2 = 0.3
    # Get Original L in LAB, shift H in HSV

    # Get Original LAB
    lab_original = color.rgb2lab(I)
    l_original = (lab_original[:, :, 0] / 100.0)
    
    # Shift HSV
    hsv = color.rgb2hsv(I)
    h = ((hsv[:, :, 0] + shift))
    s = (hsv[:, :, 1]) * shift2
    v = (hsv[:, :, 2])
    hsv2 = color.hsv2rgb(np.dstack([h, s, v]).astype(np.float64))

    # Merge (Original LAB, Shifted HSV)
    lab = color.rgb2lab(hsv2)
    l = l_original # -1 to 1
    #l = (lab[:, :, 0] / 100.0)
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -1 to 1
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -1 to 1

    return np.dstack([l, a, b])

def RGB2HSV_shift2LAB_cv(bgr, type = 'lab'):
    # min for HSV is 0
    H_max = 360.
    S_max = 1.
    V_max = 1.

    l_max = 100.
    l_min = 0.0
    a_max = 98.2330538631
    a_min = -86.1830297444
    b_max = 94.4781222765
    b_min = -107.857300207

    shift = random.uniform(0, 1)
    shift2 = random.uniform(0.8, 1.2)
    # shift3 = random.uniform(0.8, 1.2)

    # BGR2HSV and shift HSV
    # lab_original = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2LAB)

    hsv = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2HSV)

    h = (hsv[:, :, 0] + H_max * shift) % H_max
    # h = hsv[:, :, 0]
    # s = hsv[:, :, 1] * shift2
    s = hsv[:, :, 1]
    s[np.where(s > S_max)] = S_max
    v = hsv[:, :, 2]
    hsv = np.dstack([h, s, v])

    # HSV2LAB
    rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)

    # print('mean_a: ', np.mean(lab[:, :, 1]))
    # print('std_a: ', np.std(lab[:, :, 1]))
    # print('mean_b: ', np.mean(lab[:, :, 2]))
    # print('std_b: ', np.std(lab[:, :, 2]))

    l = (lab[:, :, 0] / l_max) 
    a = (lab[:, :, 1] - a_min) / (a_max - a_min) # 0 to 1
    b = (lab[:, :, 2] - b_min) / (b_max - b_min) # 0 to 1
    lab = np.dstack([l, a, b])

    # -1 to 1
    if type == 'lab':
        return (lab - 0.5) / 0.5, None
    elif type == 'rgb':
        return (lab - 0.5) / 0.5, (rgb * 2.) - 1.

def RGB2HSV_shift2HSV_cv(bgr):
    # min for HSV is 0
    H_max = 360.
    S_max = 1.
    V_max = 1.

    shift = random.uniform(0, 1)
    shift2 = random.uniform(0.8, 1.2)

    hsv = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2HSV)

    h = ((hsv[:, :, 0] + H_max * shift) % H_max) / 360.
    s = hsv[:, :, 1] * shift2
    s[np.where(s > S_max)] = S_max
    v = hsv[:, :, 2]
    hsv = np.dstack([h, s, v])

    return (hsv * 2) - 1

    

def RGB2LAB_cv(bgr, type = 'lab'):
    # min for HSV is 0
    l_max = 100.
    l_min = 0.0
    a_max = 98.2330538631
    a_min = -86.1830297444
    b_max = 94.4781222765
    b_min = -107.857300207

    lab = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2LAB)

    l = (lab[:, :, 0] / l_max) 
    a = (lab[:, :, 1] - a_min) / (a_max - a_min) # 0 to 1
    b = (lab[:, :, 2] - b_min) / (b_max - b_min) # 0 to 1
    lab = np.dstack([l, a, b])

    if type == 'lab':
        return (lab - 0.5) / 0.5, None
    elif type == 'rgb':
        return (lab - 0.5) / 0.5, (bgr * 2.) - 1.

def RGB2HSV_shift2RGB_cv(bgr):
    # min for HSV is 0
    H_max = 360.
    S_max = 1.
    V_max = 1.

    shift = random.uniform(0, 1)
    shift2 = random.uniform(0.8, 1.2)
    # shift3 = random.uniform(0.8, 1.2)

    # BGR2HSV and shift HSV
    hsv = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2HSV)

    h = (hsv[:, :, 0] + H_max * shift) % H_max
    # h = hsv[:, :, 0]
    s = hsv[:, :, 1] * shift2
    # s = hsv[:, :, 1]
    s[np.where(s > S_max)] = S_max
    v = hsv[:, :, 2]
    hsv = np.dstack([h, s, v])

    # HSV2LAB
    rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)

    return (rgb * 2.) - 1.

def getHistogram3d_np(img, num_bin, include_0, is_numpy = True):
    if is_numpy is False:
        img = img.detach().cpu().numpy()
        img = img.transpose(0, 2, 3, 1)

    arr = img
    # Exclude Zeros and Make value 0 ~ 1
    if include_0:
        arr0 = (arr[0, :, :, 0].ravel() + 1) /2 
        arr1 = (arr[0, :, :, 1].ravel() + 1) /2 
        arr2 = (arr[0, :, :, 2].ravel() + 1) /2 
    else:
        arr0 = ( arr[0, :, :, 0].ravel()[np.flatnonzero(arr[0, :, :, 0])] + 1 ) /2 
        arr1 = ( arr[0, :, :, 1].ravel()[np.flatnonzero(arr[0, :, :, 1])] + 1 ) /2 
        arr2 = ( arr[0, :, :, 2].ravel()[np.flatnonzero(arr[0, :, :, 2])] + 1 ) /2 

    if (arr1.shape[0] != arr2.shape[0]):
        print('Histogram Size Not Match!: arr1: ', arr1.shape[0], ' arr2: ', arr2.shape[0])
        arr2 = np.concatenate([arr2, np.array([0])])

    # AB space
    arr_new = [arr0, arr1, arr2]
    H,edges = np.histogramdd(arr_new, bins = [num_bin, num_bin, num_bin], range = ((0,1),(0,1),(0,1)))

    # Normalize
    total_num = H.sum() # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
    H = H / total_num
    H = np.expand_dims(H, 0)

    if is_numpy is False:
        H = torch.from_numpy(H.transpose(0, 3, 1, 2))

    return H

def getHistogram2d_np(img, num_bin, include_0, axis1 = 1, axis2 = 2, is_numpy = True):
    if is_numpy is False:
        img = img.detach().cpu().numpy()
        img = img.transpose(0, 2, 3, 1)

    arr = img
    # Exclude Zeros and Make value 0 ~ 1
    if include_0:
        arr1 = (arr[0, :, :, axis1].ravel() + 1) /2 
        arr2 = (arr[0, :, :, axis2].ravel() + 1) /2 
    else:
        arr1 = ( arr[0, :, :, axis1].ravel()[np.flatnonzero(arr[0, :, :, axis1])] + 1 ) /2 
        arr2 = ( arr[0, :, :, axis2].ravel()[np.flatnonzero(arr[0, :, :, axis2])] + 1 ) /2 

    if (arr1.shape[0] != arr2.shape[0]):
        print('Histogram Size Not Match!: arr1: ', arr1.shape[0], ' arr2: ', arr2.shape[0])
        arr2 = np.concatenate([arr2, np.array([0])])

    # AB space
    arr_new = [arr1, arr2]
    H,edges = np.histogramdd(arr_new, bins = [num_bin, num_bin], range = ((0,1),(0,1)))

    H = np.rot90(H)
    H = np.flip(H,0)

    # Normalize
    total_num = H.sum() # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
    H = H / total_num
    # cv2.imwrite('H.png', H*255)
    # print(np.flatnonzero(H))
    # exit()

    H = np.expand_dims(np.expand_dims(H, 0), 3)

    if is_numpy is False:
        H = torch.from_numpy(H.transpose(0, 3, 1, 2))

    return H

def getHistogram1d_np(img, num_bin, include_0, axis = 0, is_tile = True, is_numpy = True): # L space # Idon't know why but they(np, conv) are not exactly same
    if is_numpy is False:
        img = img.detach().cpu().numpy()
        img = img.transpose(0, 2, 3, 1)

    # Preprocess
    arr = img
    if include_0:
        arr0 = ( arr[0, :, :, axis].ravel() + 1 ) / 2 
    else:
        arr0 = ( arr[0, :, :, axis].ravel()[np.flatnonzero(arr[0, :, :, axis])] + 1 ) / 2 
    arr1 = np.zeros(arr0.size)

    arr_new = [arr0, arr1]
    H, edges = np.histogramdd(arr_new, bins = [num_bin, 1], range =((0,1),(-1,2)))

    total_num = H.sum() # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
    H = H / total_num

    # print('1. ', H.shape)
    H = np.expand_dims(np.expand_dims(H, 0), 0).transpose(0, 1, 3, 2)
    # print('2. ', H.shape)
    if is_tile:
        H = np.tile(H, (1, 64, 64, 1))

    if is_numpy is False:
        H = torch.from_numpy(H.transpose(0, 3, 1, 2))

    return H

def MakeLabelFromMap(input_A_Map, input_B_Map):
    label_A = LabelFromMap(input_A_Map)
    label_B = LabelFromMap(input_B_Map)

    label_AB2 = np.concatenate((label_A,label_B), axis = 0)
    label_AB2 = np.unique(label_AB2, axis = 0)
    label_AB  = torch.from_numpy(label_AB2)

    A_seg = np.zeros((input_A_Map.shape[0],input_A_Map.shape[1], 1))
    B_seg = np.zeros((input_B_Map.shape[0],input_B_Map.shape[1], 1))

    for i in range(0,label_AB.size(0)):            
        A_seg[ (input_A_Map == np.expand_dims(np.expand_dims(label_AB[i], 0), 0))[:,:,0:1] ] = i
        B_seg[ (input_B_Map == np.expand_dims(np.expand_dims(label_AB[i], 0), 0))[:,:,0:1] ] = i

    return A_seg, B_seg, label_AB.size(0)

def LabelFromMap(tensor_map):
    # np1 = tensor_map.squeeze(0).detach().cpu().numpy()
    # np2 = np.transpose(np1, (1,2,0))
    np3 = np.reshape(tensor_map, (np.shape(tensor_map)[0] * np.shape(tensor_map)[1], 3))
    np4 = np.unique(np3, axis= 0)

    return np4    

def serep(img, seg_src, seg_tgt, final_tensor, segment_num):
    
    # Mask only Specific Segmentation
    mask_seg = torch.mul( img , (seg_src == segment_num).cuda().float() )

    #Calc Each Histogram
    with torch.no_grad():
        hist_2d = getHistogram2d_np(mask_seg, 64)
        hist_1d = getHistogram1d_np(mask_seg, 8).repeat(1,1,64,64)
        hist_cat   = torch.cat((hist_2d,hist_1d),1)

    #Encode Each Histogram Tensor
    hist_feat = self.netC_A(hist_cat)

    #Embeded to Final Tensor
    final_tensor[:,:,seg_tgt.squeeze(0)==segment_num] = hist_feat.repeat(1,1, final_tensor[:,:,seg_tgt.squeeze(0)==segment_num].size(2), 1).squeeze(0).permute(2,0,1)
