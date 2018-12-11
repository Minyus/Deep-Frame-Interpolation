import argparse
import cv2
from src.utils import VideoInterpTripletsDataset
import sys
import torch
sys.path.append("./models")

def tensor2numpy(tensor):
    t = tensor.numpy().transpose((1,2,0))
    return np.interp(output_image,(-1.0,1.0),(0,255.0)).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frames', help='path to frames')
    parser.add_argument('video', help='path to video')
    parser.add_argument('generator', help='path to generator')
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    dataset = VideoInterpTripletsDataset(args.frames, read_frames=True)
    height, width = dataset.getsize()
    gen = torch.load(args.generator, map_location='cpu')
    print('Model Architecture Generator: ')
    for name, param in gen.named_parameters():
        if param.requires_grad:
            print(name)
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        gen = gen.cuda()
        dtype = torch.cuda.FloatTensor
    gen.eval()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))
    with torch.no_grad():
        for i in range(0, len(dataset) - 2, 2):
            data = {f:dataset[i][f][None, :] for f in dataset[i]}
            inframes = (data['left'], data['right'])
            g = gen(inframes)
            outframe = tensor2numpy(g)
            if i == 0:
                out.write(tensor2numpy(inframes[0]))
            out.write(outframe)
            out.write(tensor2numpy(inframes[1]))
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
