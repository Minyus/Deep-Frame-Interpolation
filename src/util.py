import glob
import numpy as np
import os
from skimage.transform import resize
import skvideo.io
import torch
from torch.utils.data import Dataset, DataLoader

class VideoInterpTripletsDataset(torch.utils.data.Dataset):
    def __init__(self, dir, height=144, width=256, transform=None):
        """
        :param dir: directory of videos
        :param height = pixel height of frame
        :param width = pixel width of frame
        :param transform:
        """
        self.filenames = [filename for filename in glob.glob(os.path.join(videoDirectory,'*.mp4'))]
        self.height = height
        self.width = width
        self.transform = transform
        self.frames = [int(skvideo.io.ffprobe(f)['video']['@nb_frames']) for f in self.filenames]
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
        # if index near end of file, make sure it has enough frames to read
        index = min(index, self.frames[file] - 3)

        reader = skvideo.io.vreader(self.filenames[file], inputdict={'--start_number':str(index),'-vframes':str(3)})
        triplet = [resize(frame, (self.height, self.width)) for frame in reader]
        triplet = [np.transpose(frame, (2, 1, 0))[None] for frame in triplet] # (1, C, H, W)
        inframes = np.stack([triplet[0], triplet[2]], axis=-1) # (1, C, H, W, 2)
        return inframes, triplet[1]

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