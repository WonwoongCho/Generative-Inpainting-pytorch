# Generative-Inpainting-pytorch
This is a pytorch version of the paper 'Generative Image Inpainting with Contextual Attention' (by Jiahui Yu et al. / 2018 CVPR)

### Dependencies
> python 3.6<br />
> pytorch 0.3.0

### Link
> **paper:** https://arxiv.org/abs/1801.07892 <br />
> **original TF code**: https://github.com/JiahuiYu/generative_inpainting

### Data
I used celebA faces dataset. (about 202,000 images)<br />
place it in the 'data' directory
> images=https://www.dropbox.com/s/3e5cmqgplchz85o/CelebA_nocrop.zip?dl=0 <br />
> attributes: https://www.dropbox.com/s/auexdy98c6g7y25/list_attr_celeba.zip?dl=0 (just for getting file names)<br />

### Running code
> python run.py

### P.S
I made this network for catching up SOTA technique of the Image Inpainting area.<br />
Though it trains without any bug, generated images are mostly blurry.<br />

**After clonning the repo, you need to check the contextual module performance or critics convergence.**<br />
<br />
