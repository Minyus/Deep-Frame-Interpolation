import argparse
import cv2
from models.UNet_model import UNetGenerator
import numpy as np
from src.utils import VideoInterpTripletsDataset
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import torch
sys.path.append("./models")

def write2video(tensor, vid):
    t = tensor.numpy().transpose((1,2,0))
    npt = np.interp(t, (-1.0,1.0), (0,255.0)).astype(np.uint8)
    cvt = cv2.cvtColor(npt, cv2.COLOR_RGB2BGR)
    vid.write(cvt)

def mix_frames(real, gen):
    width = real.shape[-1]
    half = int(width / 2)
    if not real.equal(gen):
        gen[:, :, :, :half] = real[:, :, :, :half]
    gen[:, :, :, half : half + 2] = 1
    return gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='one of halfgen, fullgen, double, slomo', default='halfgen')
    parser.add_argument('frames', help='path to frames')
    parser.add_argument('video', help='path to video')
    parser.add_argument('generator', help='path to generator')
    # parser.add_argument('mix', help='mix output between two frames', default=True)
    args = parser.parse_args()

    vidfile = args.video
    filename = vidfile[vidfile.rfind('/') + 1:vidfile.find('.mp4')]
    cam = cv2.VideoCapture(args.video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    dataset = VideoInterpTripletsDataset(args.frames, read_frames=True)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
    height, width = dataset.getsize()
#     gen = torch.load(args.generator, map_location='cpu')
#     torch.save(gen.module.state_dict(), 'model')
    gen = UNetGenerator()
    gen.load_state_dict(torch.load('model'))
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        gen = gen.cuda()
        dtype = torch.cuda.FloatTensor
    gen.eval()
    outfile = '{}_half.mp4'.format(filename)
    if os.path.exists(outfile): 
        os.remove(outfile)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    with torch.no_grad():
        for index, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            if args.mode == 'halfgen':
                sample = {k:v[::2, :, :, :] for k,v in sample.items()}
            left, right, real = sample['left'], sample['right'], sample['out']
            inframes = (left.type(dtype), right.type(dtype))
            g = gen(inframes).cpu()
            gmix = mix_frames(real, g)
            rmix = mix_frames(right, right)
            if index == 0:
                write2video(mix_frames(left[0][None,:], left[0][None,:])[0], out)
            for i in range(gmix.shape[0]):
                write2video(gmix[i], out)
                write2video(rmix[i], out)
#                 pbar.update(1)
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
