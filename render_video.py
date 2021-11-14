"""Script to render a video using a trained pi-GAN  model."""

import argparse
import math
import os

from torchvision.utils import save_image

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
from siren import siren
import skvideo.io
import curriculums
from generators import generators
import cv2 as cv
from icecream import ic

from torch_ema import ExponentialMovingAverage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seeds', nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument('--output_dir', type=str, default='vids')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=600000)
parser.add_argument('--depth_map', action='store_true')
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=36)
parser.add_argument('--curriculum', type=str, default='CelebA')
parser.add_argument('--trajectory', type=str, default='front')
opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok=True)

curriculum = getattr(curriculums, opt.curriculum)
curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
curriculum['img_size'] = opt.image_size
curriculum['psi'] = 0.7
curriculum['v_stddev'] = 0
curriculum['h_stddev'] = 0
curriculum['lock_view_dependence'] = opt.lock_view_dependence
curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


metadata = curriculums.extract_metadata(curriculum, 0)
SIREN = getattr(siren, metadata['model'])
generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
generator.load_state_dict(torch.load(opt.path, map_location=device))
ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema.load_state_dict(torch.load(ema_file, map_location=device))
# ema = ExponentialMovingAverage()
# ema = torch.load(ema_file, map_location=device)
# for key in ema:
#     ic(key)
# exit()
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

if opt.trajectory == 'front':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi + math.pi) + math.pi/2
        yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi/2
        fov = 12

        # fov = 12 + 5 + np.sin(t * 2 * math.pi) * 5

        trajectory.append((pitch, yaw, fov))
elif opt.trajectory == 'orbit':
    trajectory = []
    for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = math.pi/4
        yaw = t * 2 * math.pi
        fov = curriculum['fov']

        trajectory.append((pitch, yaw, fov))

for seed in opt.seeds:
    frames = []
    depths = []
    save_dir = os.path.join(opt.output_dir, str(seed))
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(seed)
    z = torch.randn(1, 256, device=device)

    with torch.no_grad():
        for pitch, yaw, fov in tqdm(trajectory):
            curriculum['h_mean'] = yaw
            curriculum['v_mean'] = pitch
            curriculum['fov'] = fov
            curriculum['h_stddev'] = 0
            curriculum['v_stddev'] = 0

            frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
            frames.append(((frame.permute(0, 2, 3, 1).squeeze() + 1.0) * 127.5).cpu().numpy().clip(0, 255).astype(np.uint8))

        for i, frame in enumerate(frames):
            cv.imwrite(os.path.join(save_dir, '{:0>3d}.png'.format(i)), frame[:,:,::-1])

