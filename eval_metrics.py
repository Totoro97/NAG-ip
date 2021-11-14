import os
import shutil
import torch
import math
import numpy as np
import cv2 as cv
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from tqdm import tqdm
import copy
import argparse
import shutil

import curriculums

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_file', type=str)
    parser.add_argument('--output_dir', type=str, default='eval_images')
    parser.add_argument('--num_images', type=int, default=100000)
    parser.add_argument('--max_batch_size', type=int, default=512000)
    parser.add_argument('--curriculum', type=str, default='COMPCARS')
    parser.add_argument('--start', type=int, required=True)

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if os.path.exists(opt.output_dir) and os.path.isdir(opt.output_dir):
    #     shutil.rmtree(opt.output_dir)
    
    os.makedirs(opt.output_dir, exist_ok=True)

    generator = torch.load(opt.generator_file, map_location=device)
    generator.set_device(device)
    ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.eval()
    
    curriculum = curriculums.extract_metadata(getattr(curriculums, opt.curriculum), generator.step)
    curriculum['img_size'] = 256
    curriculum['psi'] = 1
    curriculum['last_back'] = curriculum.get('eval_last_back', False)  
    curriculum['nerf_noise'] = 0
    
    for img_counter in tqdm(range(opt.start, opt.num_images)):
        z = torch.randn(1, 256, device=device)

        with torch.no_grad():
            img = generator.staged_forward(z, max_batch_size=opt.max_batch_size, **curriculum)[0].to(device)
            img = ((img + 1.0) * 127.5).permute(0, 2, 3, 1).squeeze().detach().cpu().numpy().astype(np.uint8)
            cv.imwrite(os.path.join(opt.output_dir, f'{img_counter:0>5}.jpg'), img[:, :, ::-1])
            # save_image(img, os.path.join(opt.output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))

    # metrics_dict = calculate_metrics(opt.output_dir, opt.real_image_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
    # print(metrics_dict)
