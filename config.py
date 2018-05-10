import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='')

#Image setting
image_size = 128
mask_size = int(image_size/2)

parser.add_argument('--GPU', type=int, default=0)
parser.add_argument('--CROP_SIZE', type=int, default=image_size+128) # 178
parser.add_argument('--IMG_SIZE', type=int, default=image_size)

parser.add_argument('--G_LR', type=float, default=1e-4)
parser.add_argument('--D_LR', type=float, default=1e-4)
parser.add_argument('--GLOBAL_WGAN_LOSS_ALPHA', type=float, default=1.)
parser.add_argument('--WGAN_GP_LAMBDA', type=float, default=10)
parser.add_argument('--GAN_LOSS_ALPHA', type=float, default=0.001)
parser.add_argument('--COARSE_L1_ALPHA', type=float, default=2)
parser.add_argument('--L1_LOSS_ALPHA', type=float, default=2)
parser.add_argument('--AE_LOSS_ALPHA', type=float, default=2)
parser.add_argument('--D_TRAIN_REPEAT', type=int, default=5)

# Training settings
parser.add_argument('--DATASET', type=str, default='CelebA', choices=['CelebA'])
parser.add_argument('--NUM_EPOCHS', type=int, default=100)
parser.add_argument('--NUM_EPOCHS_DECAY', type=int, default=10)
parser.add_argument('--NUM_ITERS', type=int, default=200000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=100000)
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--BETA1', type=float, default=0.5)
parser.add_argument('--BETA2', type=float, default=0.9)
parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
parser.add_argument('--SPATIAL_DISCOUNTING_GAMMA', type=float, default=0.9)

# Test settings
parser.add_argument('--TEST_MODEL', type=str, default='20_1000')

# Misc
parser.add_argument('--MODE', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--USE_TENSORBOARD', type=str2bool, default=False)

# Path
parser.add_argument('--IMAGE_PATH', type=str, default='./data/CelebA/images')
parser.add_argument('--METADATA_PATH', type=str, default='./data/list_attr_celeba.txt')
parser.add_argument('--LOG_PATH', type=str, default='logs')
parser.add_argument('--MODEL_SAVE_PATH', type=str, default='models')
parser.add_argument('--SAMPLE_PATH', type=str, default='samples')

# Step size
parser.add_argument('--PRINT_EVERY', type=int, default=400)
parser.add_argument('--SAMPLE_STEP', type=int, default=400)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=400)

# etc
parser.add_argument('--IMG_SHAPE', type=list, default=[image_size,image_size,3])
parser.add_argument('--MASK_HEIGHT', type=int, default=mask_size)
parser.add_argument('--MASK_WIDTH', type=int, default=mask_size)
parser.add_argument('--VERTICAL_MARGIN', type=int, default=0)
parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=0)
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=int(image_size/8))
parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=int(image_size/8))
args = parser.parse_args(args=[])