import argparse
import cv2
from models.UNet_model import UNetGenerator
import numpy as np
from src.utils import VideoInterpTripletsDataset
import sys
import torch
sys.path.append("./models")

def write2video(tensor, vid):
    t = tensor.squeeze().numpy().transpose((1,2,0))
    npt = np.interp(t, (-1.0,1.0), (0,255.0)).astype(np.uint8)
    cvt = cv2.cvtColor(npt, cv2.COLOR_RGB2BGR)
    vid.write(cvt)

def mix_frames(real, gen):
    width = real.shape[-1]
    gen[:, :, :, width / 2] = real[:, :, :, width / 2]
    gen[:, :, :, width / 2 : width / 2 + 2] = -1
    return gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames', help='path to frames')
    parser.add_argument('video', help='path to video')
    parser.add_argument('generator', help='path to generator')
    # parser.add_argument('mix', help='mix output between two frames', default=True)
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    dataset = VideoInterpTripletsDataset(args.frames, read_frames=True)
    height, width = dataset.getsize()
    gen = torch.load(args.generator, map_location='cpu')
    torch.save(gen.module.state_dict(), 'model')
    gen = UNetGenerator()
    gen.load_state_dict(torch.load('model'))
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        gen = gen.cuda()
        dtype = torch.cuda.FloatTensor
    gen.eval()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))
    with torch.no_grad():
        for i in range(0, len(dataset) - 2, 2):
            if i % 100 == 0:
                print(i)
            data = {f:dataset[i][f][None, :].type(dtype) for f in dataset[i]}
            inframes = (data['left'], data['right'])
            if i == 0:
                write2video(inframes[0], out)
            g = gen(inframes)
            write2video(mix_frames(data['out'], g), out)
            write2video(inframes[1], out)
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
