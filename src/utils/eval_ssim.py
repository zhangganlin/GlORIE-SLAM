import os
import shutil
import torch
import cv2
import traceback
import numpy as np
import glob

# adapted from pytorch-ssim https://github.com/Po-Hsun-Su/pytorch-ssim

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from tqdm import tqdm


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)




device = "cuda:0"
scene = "office4"

os.system(f'unzip -n /cluster/work/cvl/esandstroem/data/Replica/{scene}.zip -d $TMPDIR')

rendered_folder = f"/cluster/work/cvl/esandstroem/src/mono_point_slam/output/Replica/current_version_results/{scene}/rendered_every_keyframe"
render_img_path = sorted(glob.glob(f'{rendered_folder}/color*.npy'))
render_depth_path = sorted(glob.glob(f'{rendered_folder}/depth*.npy'))

from src.utils.datasets import get_dataset
from src import config
cfg = config.load_config(
        f'./configs/Replica/{scene}.yaml', './configs/mono_point_slam.yaml'
)
dataset = get_dataset(cfg, device=device)


ssim_computer = SSIM()

total_ssim = 0
total_img = len(render_img_path)

for i in tqdm(range(total_img)):
    img_path = render_img_path[i]
    depth_path = render_depth_path[i]
    idx = int(img_path[-9:-4])
    render_img = np.load(img_path) #[H,W,3]
    render_depth = np.load(depth_path)
    
    gt_img = dataset.get_color(idx).to(device)
    render_img = torch.from_numpy(render_img).to(
                 device).permute(2,0,1).unsqueeze(0)
    render_depth = torch.from_numpy(render_depth).to(device)
    valid_mask = (render_depth!=0)

    render_img[:,:,~valid_mask]=0
    gt_img[:,:,~valid_mask]=0

    ssim_value = ssim_computer(render_img, gt_img)
    total_ssim += ssim_value
    # print(render_img.shape)
    # print(gt_img.shape)
    # print(valid_mask.shape)
print(f"{scene} avg ssim: {total_ssim/total_img}")
