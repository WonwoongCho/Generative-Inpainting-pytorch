import torch as nn
import torch.optim as optim
from torch import cuda
from config import *
from model import *
from data_loader import *
from util import *
import time
import datetime
import os
from torchvision.utils import save_image

class Run(object):
    def __init__(self, args):
        # Data loader
        if args.DATASET == 'CelebA':
            self.data_loader = get_loader(args.IMAGE_PATH, args.METADATA_PATH, 
                                        args.CROP_SIZE, args.IMG_SIZE, 
                                        args.BATCH_SIZE, args.DATASET, args.MODE)

        # Model hyper-parameters
        self.image_shape = args.IMG_SHAPE

        # Hyper-parameteres
        self.stage1_lambda_l1 = args.COARSE_L1_ALPHA
        self.global_wgan_loss_alpha = args.GLOBAL_WGAN_LOSS_ALPHA
        self.wgan_gp_lambda = args.WGAN_GP_LAMBDA
        self.gan_loss_alpha = args.GAN_LOSS_ALPHA
        self.l1_loss_alpha = args.L1_LOSS_ALPHA
        self.ae_loss_alpha = args.AE_LOSS_ALPHA
        self.g_lr = args.G_LR
        self.d_lr = args.D_LR
        self.beta1 = args.BETA1
        self.beta2 = args.BETA2

        # Training settings
        self.dataset = args.DATASET
        self.num_epochs = args.NUM_EPOCHS
        self.num_epochs_decay = args.NUM_EPOCHS_DECAY
        self.num_iters = args.NUM_ITERS
        self.num_iters_decay = args.NUM_ITERS_DECAY
        self.batch_size = args.BATCH_SIZE
        self.use_tensorboard = args.USE_TENSORBOARD
        self.pretrained_model = args.PRETRAINED_MODEL
        self.d_train_repeat = args.D_TRAIN_REPEAT

        # Test settings
        self.test_model = args.TEST_MODEL

        # Path
        self.sample_path = args.SAMPLE_PATH
        self.model_save_path = args.MODEL_SAVE_PATH

        # Step size
        self.print_every = args.PRINT_EVERY
        self.sample_step = args.SAMPLE_STEP
        self.model_save_step = args.MODEL_SAVE_STEP

        # etc
        self.make_dir()
        self.init_network(args)
        self.loss = {}

        if self.pretrained_model:
            self.load_pretrained_model()

    def make_dir(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)

    def init_network(self, args):

        # Models
        self.G = Generator()
        self.D = Discriminator()

        # Optimizers
        self.g_optimizer = optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Loss
        self.L1 = Discounted_L1(args)
        self.torch_L1 = nn.L1Loss()

        # etc.
        self.util = Util(args)

        # Print networks
        # self.util.print_network(self.G, 'G')
        # self.util.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.L1 = self.L1.cuda()
            self.torch_L1 = nn.L1Loss().cuda()

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'G_{}_L1_{}.pth'.format(self.pretrained_model, self.l1_loss_alpha))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'D_{}_L1_{}.pth'.format(self.pretrained_model, self.l1_loss_alpha))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def train(self):   

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        start_time = time.time()
        self.G.train()
        self.D.train()
        for epoch in range(start, self.num_epochs):
            for batch, real_image in enumerate(self.data_loader): # real_image : B x 3 x H x W
                
                batch_size = real_image.size(0)
                real_image = 2.*real_image - 1. # [-1,1]
                
                # one bbox for each batch, ( top, left, maxH, maxW )
                # W and H will be reduced at the function bbox2mask
                bbox = self.util.random_bbox()

                binary_mask = self.util.bbox2mask(bbox)
                inverse_mask = 1.- binary_mask
                masked_image = real_image.clone()*inverse_mask

                binary_mask = to_var(binary_mask)
                inverse_mask = to_var(inverse_mask)
                masked_image = to_var(masked_image)
                real_image = to_var(real_image)

                stage_1, stage_2, offset_flow = self.G(masked_image, binary_mask)
                
                fake_image = stage_2*binary_mask + masked_image*inverse_mask # mask_location: generated, around_mask: ground_truth

                real_patch = self.util.local_patch(real_image, bbox)
                stage_1_patch = self.util.local_patch(stage_1, bbox)
                stage_2_patch = self.util.local_patch(stage_2, bbox)
                mask_patch = self.util.local_patch(binary_mask, bbox)
                fake_patch = self.util.local_patch(fake_image, bbox)

                l1_alpha = self.stage1_lambda_l1
                self.loss['recon'] = l1_alpha * self.L1(stage_1_patch, real_patch) # Coarse Network reconstruction loss
                self.loss['recon'] = self.loss['recon'] + self.L1(stage_2_patch, real_patch) # Refinement Network reconstruction loss
                
                self.loss['ae_loss'] = l1_alpha * self.torch_L1(stage_1*inverse_mask, real_image*inverse_mask) # recon loss except mask
                self.loss['ae_loss'] = self.loss['ae_loss'] + self.torch_L1(stage_2*inverse_mask, real_image*inverse_mask) # recon loss except mask
                self.loss['ae_loss'] = self.loss['ae_loss'] / torch.mean(torch.mean(inverse_mask, dim=3), dim=2) # 1 x 1 tensor

                if (batch+1) % self.d_train_repeat == 0:
                    global_real_fake_image = torch.cat([real_image, fake_image], dim=0)
                    local_real_fake_image = torch.cat([real_patch, fake_patch], dim=0)
                else:
                    global_real_fake_image = torch.cat([real_image, fake_image.clone()], dim=0)
                    local_real_fake_image = torch.cat([real_patch, fake_patch.clone()], dim=0)

                global_real_fake_vector, local_real_fake_vector = self.D(global_real_fake_image, local_real_fake_image)

                global_real_vector, global_fake_vector = torch.split(global_real_fake_vector, batch_size, dim=0)
                local_real_vector, local_fake_vector = torch.split(local_real_fake_vector, batch_size, dim=0)

                global_G_loss, global_D_loss = self.wgan_loss(global_real_vector, global_fake_vector)
                local_G_loss, local_D_loss = self.wgan_loss(local_real_vector, local_fake_vector)

                self.loss['g_loss'] = self.global_wgan_loss_alpha * (global_G_loss + local_G_loss)
                self.loss['d_loss'] = global_D_loss + local_D_loss

                if (batch+1) % self.d_train_repeat == 0:
                    # gradient penalty
                    global_interpolate = self.random_interpolates(real_image, fake_image) 
                    local_interpolate = self.random_interpolates(real_patch, fake_patch)
                else:
                    global_interpolate = self.random_interpolates(real_image, fake_image.clone()) 
                    local_interpolate = self.random_interpolates(real_patch, fake_patch.clone())

                global_gp_vector, local_gp_vector = self.D(global_interpolate, local_interpolate)

                global_penalty = self.gradient_penalty(global_interpolate, global_gp_vector, mask=binary_mask)
                local_penalty = self.gradient_penalty(local_interpolate, local_gp_vector, mask=mask_patch)

                self.loss['gp_loss'] = self.wgan_gp_lambda * (local_penalty + global_penalty)
                self.loss['d_loss'] = self.loss['d_loss'] + self.loss['gp_loss']

                if (batch+1) % self.d_train_repeat == 0:             
                    self.loss['g_loss'] = self.gan_loss_alpha * self.loss['g_loss']
                    self.loss['g_loss'] = self.loss['g_loss'] + self.l1_loss_alpha * self.loss['recon'] + self.ae_loss_alpha * self.loss['ae_loss']
                    self.backprop(D=True,G=True)

                else:                  
                    self.loss['g_loss'] = to_var(torch.FloatTensor([0]))
                    self.backprop(D=True,G=False)


                if batch % self.print_every == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    print('=====================================================')
                    print("Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, epoch+1, self.num_epochs, batch+1, iters_per_epoch))
                    print('=====================================================')
                    print('reconstruction loss: ', self.loss['recon'].data[0])
                    print('ae loss: ', self.loss['ae_loss'].data[0][0])
                    print('g loss: ', self.loss['g_loss'].data[0])
                    print('d loss: ', self.loss['d_loss'].data[0])
                    show_image(real_image, (masked_image+binary_mask), stage_1, stage_2, fake_image, offset_flow)

                # Save model checkpoints
                if batch % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, 'G_{}_L1_{}.pth'.format(epoch+1, self.l1_loss_alpha)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, 'D_{}_L1_{}.pth'.format(epoch+1, self.l1_loss_alpha)))

                # Save sample image
                if batch % self.sample_step == 0:
                    save_image(self.denorm(fake_image.clone().data.cpu()),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(epoch+1, batch+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

    def backprop(self, D=True, G=True):
        if D:
            self.d_optimizer.zero_grad()
            self.loss['d_loss'].backward(retain_graph=G)
            self.d_optimizer.step()
        if G:
            self.g_optimizer.zero_grad()
            self.loss['g_loss'].backward()
            self.g_optimizer.step()

    def wgan_loss(self, real, fake):
        diff = fake - real
        d_loss = torch.mean(diff)
        g_loss = -torch.mean(fake)
        return g_loss, d_loss

    def random_interpolates(self, real, fake, alpha=None):
        shape = list(real.size())
        real = real.contiguous().view(shape[0], -1, 1, 1)
        fake = fake.contiguous().view(shape[0], -1, 1, 1)
        if alpha is None:
            alpha = Variable(torch.rand(shape[0], 1, 1, 1)).cuda()
        interpolates = fake + alpha*(real - fake)
        return interpolates.view(shape)

    def gradient_penalty(self, x, y, mask=None, norm=1.):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = Variable(torch.ones(y.size())).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx * mask
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

def main(_):

    cuda.set_device(args.GPU)
    print("Running on GPU : ", args.GPU)
    run = Run(args)

    if args.MODE == 'train':
        run.train()
    else:
        run.test()

main(args)