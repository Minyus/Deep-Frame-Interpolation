import glob
import numpy as np
import os
from skimage.transform import resize
import skvideo.io
import torch
from torch.utils.data import Dataset

class VideoInterpTripletsDataset(Dataset):
    def __init__(self, directory, height=144, width=256, transform=None):
        """
        :param directory: directory of videos
        :param height = pixel height of frame
        :param width = pixel width of frame
        :param transform:
        """
        self.filenames = [filename for filename in glob.glob(os.path.join(directory,'*.mp4'))]
        self.height = height
        self.width = width
        self.transform = transform # TODO(wizeng): Implement crop, tensor, and resize transforms
        self.frames = [int(skvideo.io.ffprobe(f)['video']['@nb_frames']) - 2 for f in self.filenames]
        self.total = sum(self.frames)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        '''
        :param index: idx of the first frame in the video you want
        :return: (inframes, outframes)((1,C,H,W,2),(1,C,H,W))
        '''

        # iterate through video's frame lengths to the correct video
        file = 0
        while self.frames[file] <= index:
            index -= self.frames[file]
            file += 1
        print(self.filenames[file])
        print(self.total)
        print(index)
        reader = skvideo.io.vreader(self.filenames[file], inputdict={'--start_number':str(index),'-vframes':str(3)})
        triplet = []
        print('Lol')
        for frame in reader:
            print('Hi')
            triplet.append(resize(frame, (self.height, self.width)))
        print('poo')

        triplet = [torch.from_numpy(frame.transpose((2, 1, 0))) for frame in triplet] # (C, H, W)
        print(len(triplet))
        return {'left': triplet[0], 'right': triplet[2], 'out': triplet[1]}

        # totalLen = 0
        # correctFilename = None
        # for filename in self.videoFilenames:
        #    lengthOfVideo = int(skvideo.io.ffprobe(filename)['video']['@nb_frames'])
        #    if totalLen+lengthOfVideo-3 >= index:
        #        correctFilename = filename
        #        print(filename)
        #        break
        #    elif totalLen+lengthOfVideo-3 < index and totalLen + lengthOfVideo > index: #weird case
        #        correctFilename = filename
        #        # index - the amount needed to make sure you have at least 3 frames in triplet
        #        index = index - (totalen+lengthOfVideo-index)
        #        break
        #    elif totalLen+lengthOfVideo-3 < index:
        #        lenVideo += lengthOfVideo
        # print(correctFilename)