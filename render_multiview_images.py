import argparse
import math
import glob
import numpy as np
import sys
import os
import cv2 as cv
import torch
from siren import siren
from generators import generators
from torchvision.utils import save_image
from tqdm import tqdm

from torch_ema import ExponentialMovingAverage

import curriculums

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()
    
def generate_img(gen, z, **kwargs):
    with torch.no_grad():
        img, depth_map, cam2world = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()
        
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map, cam2world


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='./pretrained/afhq/generator.pth')
    parser.add_argument('--max_seed', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='../ViewConsistencyEval/pi-GAN_afhq')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    parser.add_argument('--range_u', type=float, default=0.3)
    parser.add_argument('--n_steps', type=int, default=9)
    parser.add_argument('--test_hold_out', type=int, default=2)
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    
    os.makedirs(opt.output_dir, exist_ok=True)

    metadata = curriculums.extract_metadata(curriculum, 0)
    SIREN = getattr(siren, metadata['model'])
    generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
    generator.load_state_dict(torch.load(opt.path, map_location=device))
    # generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'

    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(ema_file, map_location=device))
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    face_angles = np.linspace(-opt.range_u, opt.range_u, opt.n_steps) * -1

    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    for seed in tqdm(range(opt.max_seed)):
        curr_out_dir = os.path.join(opt.output_dir, 'seed_{:0>6d}'.format(seed))
        os.makedirs(curr_out_dir, exist_ok=True)
        img_dir = os.path.join(curr_out_dir, 'images_raw')
        os.makedirs(img_dir, exist_ok=True)

        images = []
        intrinsics = []
        poses = []

        for i, yaw in enumerate(face_angles):
            curriculum['h_mean'] = yaw
            torch.manual_seed(seed)
            z = torch.randn((1, 256), device=device)
            img, tensor_img, depth_map, cam2world = generate_img(generator, z, **curriculum)
            images.append(tensor_img)
            H, W, _ = img.shape
            fov = curriculum['fov']
            focal = (H - 1) * 0.5 / np.tan(fov * 0.5 / 180 * np.pi)
            intri = np.diag([focal, focal, 1.0, 1.0]).astype(np.float32)
            intri[0, 2], intri[1, 2] = (W - 1) * 0.5, (H - 1) * 0.5
            pose = cam2world.squeeze().detach().cpu().numpy() @ np.diag([1, -1, -1, 1]).astype(np.float32)
            intrinsics.append(intri)
            poses.append(pose)
            cv.imwrite(os.path.join(img_dir, '{:0>3d}.png'.format(i)), img[:,:,::-1] * 255.0)

        intrinsics = np.stack(intrinsics, axis=0)
        poses = np.stack(poses, axis=0)
        np.savez(os.path.join(curr_out_dir, 'cameras.npz'), intrinsics=intrinsics, poses=poses)
        with open(os.path.join(curr_out_dir, 'meta.conf'), 'w') as f:
            f.write('depth_range = {}\ntest_hold_out = {}\nheight = {}\nwidth = {}'.
                    format([curriculum['ray_start'], curriculum['ray_end']], opt.test_hold_out, H, W))
