import torch
import torch.nn as nn
from model_module import *
from torch.autograd import Variable
from util import *

class Generator(nn.Module):
    def __init__(self, first_dim=32):
        super(Generator,self).__init__()
        self.stage_1 = CoarseNet(5, first_dim)
        self.stage_2 = RefinementNet(5, first_dim)

    def forward(self, masked_img, mask): # mask : 1 x 1 x H x W
        # border, maybe
        mask = mask.expand(masked_img.size(0),1,masked_img.size(2),masked_img.size(3))
        ones = to_var(torch.ones(mask.size()))

        # stage1
        stage1_input = torch.cat([masked_img, ones, ones*mask], dim=1)
        stage1_output, resized_mask = self.stage_1(stage1_input, mask[0].unsqueeze(0))

        # stage2
        new_masked_img = stage1_output*mask + masked_img.clone()*(1.-mask)
        stage2_input = torch.cat([new_masked_img, ones, ones*mask], dim=1)
        stage2_output, offset_flow = self.stage_2(stage2_input, resized_mask[0].unsqueeze(0))

        return stage1_output, stage2_output, offset_flow


class CoarseNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''
    def __init__(self, in_ch, out_ch):
        super(CoarseNet,self).__init__()
        self.down = Down_Module(in_ch, out_ch)
        self.atrous = Dilation_Module(out_ch*4, out_ch*4)
        self.up = Up_Module(out_ch*4, 3)

    def forward(self, x, mask):
        x = self.down(x)
        resized_mask = down_sample(mask, scale_factor=0.25, mode='nearest')
        x = self.atrous(x)
        x = self.up(x)
        return x, resized_mask


class RefinementNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''
    def __init__(self, in_ch, out_ch):
        super(RefinementNet,self).__init__()
        self.down_conv_branch = Down_Module(in_ch, out_ch)
        self.down_attn_branch = Down_Module(in_ch, out_ch, activation=nn.ReLU())
        self.atrous = Dilation_Module(out_ch*4, out_ch*4)
        self.CAttn = Contextual_Attention_Module(out_ch*4, out_ch*4)
        self.up = Up_Module(out_ch*8, 3, isRefine=True)

    def forward(self, x, resized_mask):
        # conv branch
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)
        
        # attention branch
        attn_x = self.down_attn_branch(x)
        attn_x, offset_flow = self.CAttn(attn_x, attn_x, mask=resized_mask) # attn_x => B x 128(32*4) x W/4 x H/4

        # concat two branches
        deconv_x = torch.cat([conv_x, attn_x], dim=1) # deconv_x => B x 256 x W/4 x H/4
        x = self.up(deconv_x)

        return x, offset_flow


class Discriminator(nn.Module):
    def __init__(self, first_dim=64):
        super(Discriminator,self).__init__()
        self.global_discriminator = Flatten_Module(3, first_dim, False) 
        self.local_discriminator = Flatten_Module(3, first_dim, True)

    def forward(self, global_x, local_x):
        global_y = self.global_discriminator(global_x)
        local_y = self.local_discriminator(local_x)
        return global_y, local_y # B x 256*(256 or 512)
