import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, K=3, S=1, P=1, D=1, activation=nn.ELU()):
        super(Conv, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


# conv 1~6
class Down_Module(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.ELU()):
        super(Down_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5))

        curr_dim = out_ch
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim*2, K=3, S=2))
            layers.append(Conv(curr_dim*2, curr_dim*2))
            curr_dim *= 2
        
        layers.append(Conv(curr_dim, curr_dim, activation=activation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


# conv 7~10
class Dilation_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilation_Module, self).__init__()
        layers = []
        dilation = 1
        for i in range(4):
            dilation *= 2
            layers.append(Conv(in_ch, out_ch, D=dilation, P=dilation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


# conv 11~17
class Up_Module(nn.Module):
    def __init__(self, in_ch, out_ch, isRefine=False):
        super(Up_Module, self).__init__()
        layers = []
        curr_dim = in_ch
        if isRefine:
            layers.append(Conv(curr_dim, curr_dim//2))
            curr_dim //= 2
        else:
            layers.append(Conv(curr_dim, curr_dim))
        
        # conv 12~15
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(Conv(curr_dim, curr_dim//2))
            curr_dim //= 2

        layers.append(Conv(curr_dim, curr_dim//2))
        layers.append(Conv(curr_dim//2, out_ch, activation=0))

        self.out = nn.Sequential(*layers)

    def forward(self, x):
        output = self.out(x)
        return torch.clamp(output, min=-1., max=1.)


class Flatten_Module(nn.Module):
    def __init__(self, in_ch, out_ch, isLocal=True):
        super(Flatten_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5, S=2, P=2, activation=nn.LeakyReLU()))
        curr_dim = out_ch

        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim*2, K=5, S=2, P=2, activation=nn.LeakyReLU()))
            curr_dim *= 2
        
        if isLocal:
            layers.append(Conv(curr_dim, curr_dim*2, K=5, S=2, P=2, activation=nn.LeakyReLU()))
        else:
            layers.append(Conv(curr_dim, curr_dim, K=5, S=2, P=2, activation=nn.LeakyReLU()))

        self.out = nn.Sequential(*layers)

    def forward(self, x):
        x = self.out(x)
        return x.view(x.size(0),-1) # 2B x 256*(256 or 512); front 256:16*16


# pmconv 9~10
class Contextual_Attention_Module(nn.Module):
    def __init__(self, in_ch, out_ch, rate=2, stride=1):
        super(Contextual_Attention_Module, self).__init__()
        self.rate = rate
        self.padding = nn.ZeroPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')
        layers = []
        for i in range(2):
            layers.append(Conv(in_ch, out_ch))
        self.out = nn.Sequential(*layers)

    def forward(self, f, b, mask=None, ksize=3, stride=1, 
                fuse_k=3, softmax_scale=10., training=True, fuse=True):

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

        # get shapes
        raw_fs = f.size() # B x 128 x 64 x 64
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        # extract patches from background with stride and rate
        kernel = 2*self.rate
        raw_w = self.extract_patches(b, kernel=kernel, stride=self.rate)
        raw_w = raw_w.contiguous().view(raw_int_bs[0], -1, raw_int_bs[1], kernel, kernel) # B*HW*C*K*K (B, 32*32, 128, 4, 4)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = down_sample(f, scale_factor=1/self.rate, mode='nearest')
        b = down_sample(b, scale_factor=1/self.rate, mode='nearest')
        fs = f.size() # B x 128 x 32 x 32
        int_fs = list(f.size())
        f_groups = torch.split(f, 1, dim=0) # Split tensors by batch dimension; tuple is returned

        # from b(B*H*W*C) to w(b*k*k*c*h*w)
        bs = b.size() # B x 128 x 32 x 32
        int_bs = list(b.size())
        w = self.extract_patches(b)

        w = w.contiguous().view(int_fs[0], -1, int_fs[1], ksize, ksize) # B*HW*C*K*K (B, 32*32, 128, 3, 3)

        # process mask
        if mask is not None:
            mask = down_sample(mask, scale_factor=1./self.rate, mode='nearest')
        else:
            mask = torch.zeros([1, 1, bs[2], bs[3]])

        m = self.extract_patches(mask)

        m = m.contiguous().view(1, 1, -1, ksize, ksize)  # B*C*HW*K*K
        m = m[0] # (1, 32*32, 3, 3)
        m = reduce_mean(m) # smoothing, maybe
        mm = m.eq(0.).float() # (1, 32*32, 1, 1)       
        
        w_groups = torch.split(w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = Variable(torch.eye(k).view(1, 1, k, k)).cuda() # 1 x 1 x K x K

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
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
            wi_normed = wi / torch.max(l2_norm(wi), escape_NaN)
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
            yi = yi * mm  # mm => (1, 32*32, 1, 1)
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # mask

            _, offset = torch.max(yi, dim=1) # argmax; index
            division = torch.div(offset, fs[3]).long()
            offset = torch.stack([division, torch.div(offset, fs[3])-division], dim=-1)

            # deconv for patch pasting
            # 3.1 paste center
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4. # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0) # back to the mini-batch
        y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_bs[0]] + [2] + int_bs[2:])

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
        if self.rate != 1:
            flow = self.up_sample(flow)
        return self.out(y), flow

    # padding1(16 x 128 x 64 x 64) => (16 x 128 x 64 x 64 x 3 x 3)
    def extract_patches(self, x, kernel=3, stride=1):
        x = self.padding(x)
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        return all_patches
