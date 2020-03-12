import torch
import torch.nn.functional as F
import numpy as np
import cv2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, decay_rate, decay_every, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (decay_rate ** (epoch // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def LAB2RGB_sk(I):
    I = (I * 0.5) + 0.5

    l = I[:, 0, :, :] * 100.0
    a = I[:, 1, :, :] * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, 2, :, :] * (94.4781222765 + 107.857300207) - 107.857300207
    stacked = np.stack([l, a, b], axis = 1).astype(np.float64).transpose(0, 2, 3, 1)

    rgb = color.lab2rgb(stacked[0])
    rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis = 0)
    rgb = torch.FloatTensor(rgb)

    return rgb

def LAB2RGB_cv(I, type = 'lab'):
    if type == 'lab':
        l_max = 100.
        l_min = 0.0
        a_max = 98.2330538631
        a_min = -86.1830297444
        b_max = 94.4781222765
        b_min = -107.857300207

        I = (I * 0.5) + 0.5

        l = I[:, 0, :, :] * l_max
        a = I[:, 1, :, :] * (a_max - a_min) + a_min 
        b = I[:, 2, :, :] * (b_max - b_min) + b_min
        stacked = np.stack([l, a, b], axis = 1).transpose(0, 2, 3, 1)

        rgb = cv2.cvtColor(stacked[0].astype(np.float32), cv2.COLOR_LAB2RGB)
        rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis = 0)
    elif type == 'rgb':
        rgb = (I + 1.) / 2.
    elif type == 'hsv':
        I = (I + 1.) / 2.
        I[:, 0, :, :] = I[:, 0, :, :] * 360
        I = I[0]
        I = I.transpose(1, 2, 0)
        rgb = cv2.cvtColor(I.astype(np.float32), cv2.COLOR_HSV2RGB)
        rgb = np.expand_dims(rgb.transpose(2, 0, 1), axis = 0)

    rgb = torch.FloatTensor(rgb)
    return rgb

