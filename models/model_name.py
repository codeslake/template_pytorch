import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import collections
import numpy as np

from torch_utils import *
from utils import *
from .baseModel import baseModel

class Model(baseModel):
    def __init__(self, config):
        self.config = config
        self.network = Network(config.nL + 1, config.nH, config.type, config.is_rep)

        self.network = torch.nn.DataParallel(self.network).cuda()
        self.network.module.apply(weights_init)
        self.set_optim()

    def set_loss(self, lr = None):
        print(toGreen('Building Loss...'))
        self.MSE = torch.nn.MSELoss().cuda()
        self.MAE = torch.nn.L1Loss().cuda()

    def get_results(self, inputs):
        ## network output
        self.outs['result'] = self.network(inputs['inp'])

        ## loss
        if self.config.is_train:
            self.image_loss = self.MAE if self.config.loss == 'MAE' else self.MSE
            
            self.errs['image'] = self.image_loss(self.outs['result'], inputs['gt'])
            self.errs['total'] = self.errs['image']
                             
            return self.errs, self.outs
        else:
            return self.outs


# Unet based
class Network(nn.Module):
    def __init__(self, n_in, nH, type = 'lab', is_rep = False):
        super(IRN, self).__init__()
        self.HEN = HEN(n_in, nH, type)
        self.Unet = Unet(nH)
        self.is_rep = is_rep

    def serep(self, img, seg_src, seg_tgt, final_tensor,segment_num):
        
        # Mask only Specific Segmentation
        mask_seg = torch.mul( img , (seg_src == segment_num).cuda().float() )

        #Calc Each Histogram
        with torch.no_grad():
            hist_2d = self.getHistogram2d_np(mask_seg, self.hist_ab)
            hist_1d = self.getHistogram1d_np(mask_seg, self.hist_l).repeat(1,1,self.hist_ab,self.hist_ab)
            hist_cat   = torch.cat((hist_2d,hist_1d),1)

        #Encode Each Histogram Tensor
        hist_feat = self.netC_A(hist_cat)

        #Embeded to Final Tensor
        final_tensor[:,:,seg_tgt.squeeze(0)==segment_num] = hist_feat.repeat(1,1, final_tensor[:,:,seg_tgt.squeeze(0)==segment_num].size(2), 1).squeeze(0).permute(2,0,1)

    def forward(self, inp, inp_hist, ref_hist, seg_inp = None, seg_ref = None, option = None):
        inp_H = self.HEN(inp_hist)
        ref_H = self.HEN(ref_hist)
        # print(torch.sum(torch.sum(torch.sum(inp_H - ref_H, dim = 1), dim = 1), dim = 1))

        h = inp.size()[2]
        w = inp.size()[3]
        inp_H = inp_H.repeat(1, 1, h, w)
        ref_H = ref_H.repeat(1, 1, h, w)

        if self.is_rep:
            for i in range(0,seg_num):
                seg_num_inp = torch.sum(torch.sum(self.seg_inp == i ,1),1)
                seg_num_ref = torch.sum(torch.sum(self.seg_ref == i ,1),1)
                if (seg_num_inp > 0) and (seg_num_ref > 0):
                    inp_H = self.serep(inp, seg_inp, seg_ref, inp_H, i)
                    ref_H = self.serep(ref, seg_ref, seg_inp, ref_H, i)

        out, aux_out1, aux_out2, aux_out3, aux_out4 = self.Unet(inp, inp_H, ref_H)

        return out, aux_out1, aux_out2, aux_out3, aux_out4, seg_inp, seg_ref

class HEN(nn.Module):
    def __init__(self, n_in, nH, type = 'lab', norm_layer=nn.InstanceNorm2d):
        super(HEN, self).__init__()

        self.input_nc = n_in

        self.output_nc = nH

        dim = 128

        if type == 'lab' or type == 'hsv':
            model = [
                    #### Half Size
                     nn.Conv2d(self.input_nc, dim, kernel_size=4, padding=1, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=4, padding=1, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=4, padding=1, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=4, padding=1, stride=2),
                     nn.LeakyReLU(0.1, True),
                    #### -1 Size
                     nn.Conv2d(dim, dim, kernel_size=4, padding=1, stride=1),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=4, padding=1, stride=1),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=4, padding=1, stride=1),
                     nn.LeakyReLU(0.1, True),
                    #### Out Change
                     nn.Conv2d(dim, self.output_nc, kernel_size=1, padding=0),
            ]

            self.model = nn.Sequential(*model)
            self.model2 = nn.Sequential(nn.Linear(self.output_nc, self.output_nc))

        elif type == 'rgb':
            model = [
                    #### Half Size
                     nn.Conv2d(self.input_nc, dim, kernel_size=3, padding=0, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                     nn.LeakyReLU(0.1, True),
                    #### -1 Size
                     nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=2),
                     nn.LeakyReLU(0.1, True),
                     nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=1),
                     nn.LeakyReLU(0.1, True),
                    #### Out Change
                     nn.Conv2d(dim, self.output_nc, kernel_size=1, padding=0),
            ]

            self.model = nn.Sequential(*model)
            self.model2 = nn.Sequential(nn.Linear(self.output_nc, self.output_nc))


    def forward(self, input):
        H = self.model(input)
        # print(H.size())
        H = H.view(H.size(0),-1)
        H = self.model2(H)
        H = H.unsqueeze(0).unsqueeze(0).permute(2,3,0,1) # 1/64/1/1

        return H

##############################################################################################################
#                                                   net_parts
##############################################################################################################

class Unet(nn.Module):
    def __init__(self, nH = 64):
        super(Unet, self).__init__()
        n_channels = 3
        n_classes = 3

        self.inc = inconv(n_channels, 64) # 256, 256, 64
        self.down1 = down(64 + nH*2, 128) # 128, 128, 128
        self.down2 = down(128 + nH*2, 256) # 64, 64, 256
        self.down3 = down(256 + nH*2, 512) # 32, 32, 512
        self.down4 = down(512 + nH*2, 512) # 16, 16, 512
        self.down5 = down(512 + nH*2, 512) # 16, 16, 512
        self.up0 = up(1024 + nH*2, 512)
        self.up1 = up(1024 + nH*2, 256)
        self.up2 = up(512 + nH*2, 128)
        self.up3 = up(256 + nH*2, 64)
        self.up4 = up(128 + nH*2, 64)
        self.outc = outconv(64 + nH*2, n_classes)

        self.aux_out1 = auxconv(256, 3)
        self.aux_out2 = auxconv(128, 3)
        self.aux_out3 = auxconv(64, 3)
        self.aux_out4 = auxconv(64, 3)

    def forward(self, inp, inp_H, ref_H):
        x1 = self.inc(inp)
        x2 = self.down1(x1, inp_H, ref_H)
        x3 = self.down2(x2, inp_H, ref_H)
        x4 = self.down3(x3, inp_H, ref_H)
        x5 = self.down4(x4, inp_H, ref_H)
        x6 = self.down5(x5, inp_H, ref_H)

        x = self.up0(x6, x5, inp_H, ref_H)

        x = self.up1(x, x4, inp_H, ref_H)
        aux_out1 = self.aux_out1(x)

        x = self.up2(x, x3, inp_H, ref_H)
        aux_out2 = self.aux_out2(x)

        x = self.up3(x, x2, inp_H, ref_H)
        aux_out3 = self.aux_out3(x)

        x = self.up4(x, x1, inp_H, ref_H)
        aux_out4 = self.aux_out4(x)

        x = self.outc(inp, x, inp_H, ref_H)
        # return torch.tanh(x)
        return x, aux_out1, aux_out2, aux_out3, aux_out4

class auxconv(nn.Module):
    def __init__(self, in_ch, out_ch):
      super(auxconv, self).__init__()
      
      self.conv = nn.Sequential(
          nn.Conv2d(in_ch, in_ch, kernel_size=3,stride=1, padding=1),
          nn.InstanceNorm2d(in_ch),
          nn.LeakyReLU(0.2),
          nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=1, padding=1),
          )
    
    def forward(self, feat):
        return self.conv(feat)
        
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, down = False):
        super(double_conv, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 4, stride = 2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        if self.down is False:
            x = self.conv(x)
        else:
            x = self.conv_down(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv(in_ch, out_ch, down = True)
        )

    def forward(self, x, inp_H, ref_H):
        h = x.size()[2]
        w = x.size()[3]
        # inp_H = inp_H.repeat(1, 1, h, w)
        # ref_H = ref_H.repeat(1, 1, h, w)
        inp_H = F.upsample(inp_H, size = (h, w), mode = 'bilinear')
        ref_H = F.upsample(ref_H, size = (h, w), mode = 'bilinear')

        x = torch.cat([x, inp_H, ref_H], dim=1)
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        #     self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2, inp_H, ref_H):
        x1 = self.up(x1)

        h = x1.size()[2]
        w = x1.size()[3]
        # inp_H = inp_H.repeat(1, 1, h, w)
        # ref_H = ref_H.repeat(1, 1, h, w)
        inp_H = F.upsample(inp_H, size = (h, w), mode = 'bilinear')
        ref_H = F.upsample(ref_H, size = (h, w), mode = 'bilinear')

        x = torch.cat([x1, x2, inp_H, ref_H], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        dim = 64
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2),
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2),
        )

        self.conv = nn.Conv2d(dim, out_ch, 1)


    def forward(self, inp, x, inp_H, ref_H):
        inp_ = self.conv_input(inp)

        h = inp_.size()[2]
        w = inp_.size()[3]
        # inp_H = inp_H.repeat(1, 1, h, w)
        # ref_H = ref_H.repeat(1, 1, h, w)
        inp_H = F.upsample(inp_H, size = (h, w), mode = 'bilinear')
        ref_H = F.upsample(ref_H, size = (h, w), mode = 'bilinear')


        x_res = x + inp_
        x = torch.cat([x_res, inp_H, ref_H], dim=1)
        x = self.res_block(x)

        x_res = x + x_res
        x = torch.cat([x_res, inp_H, ref_H], dim=1)
        x = self.res_block(x)

        x_res = x + x_res
        x = torch.cat([x_res, inp_H, ref_H], dim=1)
        x = self.res_block(x)

        x = self.conv(x)

        return x

class HistogramNet(nn.Module):
    def __init__(self,bin_num):
        super(HistogramNet,self).__init__()
        self.bin_num = bin_num
        self.LHConv = BiasedConv(1,bin_num)
        self.relu = nn.ReLU(True)

    def getBiasedConv(self):
        return self.LHConv

    def getBin(self):
        return self.bin_num

    def init_biased_conv(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            #m.bias.data = -torch.arange(0,1,1/(self.bin_num-1)) # Originally this was it... 19/03/02
            m.bias.data = -torch.arange(0,1,1/(self.bin_num))
            m.weight.data = torch.ones(self.bin_num,1,1,1)

    def forward(self,input):
        a1 = self.LHConv(input)
        a2 = torch.abs(a1)
        a3 = 1- a2*(self.bin_num-1)
        a4 = self.relu(a3)
        return a4


class BiasedConv(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(BiasedConv, self).__init__()
        model = []
        model += [nn.Conv2d(dim_in,dim_out,kernel_size=1,padding=0,stride=1,bias=True),]
        self.model = nn.Sequential(*model)

    def forward(self,input):
        a = self.model(input)
        return a
