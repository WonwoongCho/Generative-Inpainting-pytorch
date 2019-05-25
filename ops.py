import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
from util import *

def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=2, 
                         fuse_k=3, softmax_scale=10., training=True, fuse=True, padding=nn.ZeroPad2d(1)):

        """ Contextual attention layer implementation.

        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.

        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.

        Returns:
            tf.Tensor: output

        """
        up_sample = lambda x: F.interpolate(x, scale_factor=rate, mode='nearest')

        # get shapes
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        # extract patches from background with stride and rate
        kernel = 2*rate
        raw_w = extract_patches(padding, b, kernel=kernel, stride=rate*stride)
        raw_w = raw_w.contiguous().view(raw_int_bs[0], -1, raw_int_bs[1], kernel, kernel) # B*HW*C*K*K (B, 32*32, 128, 4, 4)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1/rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1/rate, mode='nearest')

        fs = f.size() # B x C x H x W
        int_fs = list(f.size())
        f_groups = torch.split(f, 1, dim=0) # Split tensors by batch dimension; tuple is returned

        # from b(B*H*W*C) to w(b*k*k*c*h*w)
        bs = b.size() # B x C x H x W
        int_bs = list(b.size())
        w = extract_patches(padding,b, stride=stride)
        w = w.contiguous().view(int_fs[0], -1, int_fs[1], ksize, ksize) # B*HW*C*K*K

        # process mask
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=1./rate, mode='nearest')
        else:
            mask = torch.zeros([1, 1, bs[2], bs[3]]).cuda()

        m = extract_patches(padding, mask, stride=stride)

        m = m.contiguous().view(1, -1, 1, ksize, ksize)  # B*HW*C*K*K
        #m = m[0] # (32*32, 1, 3, 3)
        m = m.mean([2,3,4]).unsqueeze(-1).unsqueeze(-1)
        mm = m.eq(0.).float() # (1, 32*32, 1, 1)       
        mm_groups = torch.split(mm, 1, dim=0)

        w_groups = torch.split(w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = Variable(torch.eye(k).view(1, 1, k, k)).cuda() # 1 x 1 x K x K
        
        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]
            escape_NaN = Variable(torch.FloatTensor([1e-4])).cuda()
            #wi_normed = wi / torch.max(l2_norm(wi), escape_NaN)
            wi_normed = wi / torch.max(torch.sqrt((wi*wi).sum([1,2,3],keepdim=True)), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=1) # yi => (B=1, C=32*32, H=32, W=32)

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                yi = yi.view(1, 1, fs[2]*fs[3], bs[2]*bs[3]) # make all of depth to spatial resolution, (B=1, I=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1) # (B=1, C=1, H=32*32, W=32*32)

                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2], bs[3]) # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, fs[2]*fs[3], bs[2]*bs[3])
                
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = yi.contiguous().view(1, fs[3], fs[2], bs[3], bs[2])
                yi = yi.permute(0, 2, 1, 4, 3)

            yi = yi.contiguous().view(1, bs[2]*bs[3], fs[2], fs[3]) # (B=1, C=32*32, H=32, W=32)

            # softmax to match
            print('yishape:',yi.shape)
            yi = yi * mi  # mi => (1, 32*32, 1, 1)
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mi

            _, offset = torch.max(yi, dim=1) # argmax; index
            division = torch.div(offset, fs[3]).long() #vertical position
            offset = torch.stack([division, torch.remainder(offset, fs[3]).long()], dim=-1) # 1 x H x W x 2

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            
            yi = F.conv_transpose2d(yi, wi_center, stride=rate, padding=1) / 4. # (B=1, C=128, H=64, W=64)

            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0) # back to the mini-batch
        y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0) #B x H x W x 2
        offsets = offsets.permute(0, 3, 1, 2) #B x 2 x H x W
        #offsets = offsets.view([int_bs[0]] + [2] + int_bs[2:])

        # case1: visualize optical flow: minus current position
        h_add = Variable(torch.arange(0,float(bs[2]))).cuda().view([1, 1, bs[2], 1])
        h_add = h_add.expand(bs[0], 1, bs[2], bs[3])
        w_add = Variable(torch.arange(0,float(bs[3]))).cuda().view([1, 1, 1, bs[3]])
        w_add = w_add.expand(bs[0], 1, bs[2], bs[3])
        offsets = offsets - torch.cat([h_add, w_add], dim=1).long()

        # to flow image
        flow = torch.from_numpy(flow_to_image(offsets.permute(0,2,3,1).cpu().data.numpy()))
        flow = flow.permute(0,3,1,2)

        # # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.int()).numpy()))
        if rate != 1:
            flow = up_sample(flow)
        return y, flow

# padding1(16 x 128 x 64 x 64) => (16 x 128 x 64 x 64 x 3 x 3)
def extract_patches(padding, x, kernel=3, stride=1):
    x = padding(x)
    x = x.permute(0, 2, 3, 1) #B x H x W x C
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches
