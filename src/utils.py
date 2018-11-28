import collections
import glob
import numpy as np
import os
from scipy import misc
import skvideo.io
import torch
from torch.utils.data import Dataset

class VideoInterpTripletsDataset(Dataset):
    def __init__(self, directory, read_frames=False):
        """
        :param directory: directory of videos
        :param read_frames: if False, read mp4's. If True, read frames with name 'videoname-framenum.jpg'
        """
        self.directory = directory
        self.read_frames = read_frames
        if self.read_frames:
            filenames = [filename for filename in glob.glob(os.path.join(directory,'*.jpg'))]
            frames = collections.defaultdict(int)
            for f in filenames:
                f = f[f.rfind('/') + 1:f.find('.jpg')]
                file = f[:f.find('-')]
                num = int(f[f.find('-') + 1 :])
                if frames[file] < num:
                    frames[file] = num
            self.filenames = [filename for filename in frames]
            self.frames = [frames[filename] - 2 for filename in self.filenames]
        else:
            self.filenames = [filename for filename in glob.glob(os.path.join(directory,'*.mp4'))]
            self.frames = [int(skvideo.io.ffprobe(f)['video']['@nb_frames']) - 2 for f in self.filenames]
        # TODO(wizeng): Implement crop, tensor, and resize transforms
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
        if self.read_frames:
            triplet = [misc.imread('{}/{}-{}.jpg'\
                .format(self.directory, self.filenames[file], i)) for i in range(index, index + 3)]
            # triplet = (np.float(triplet)/255.0) * 2.0 - 1
            triplet = [np.interp(trip,(0,255),(-1.0,1.0)) for trip in triplet]
        else:
            reader = skvideo.io.vreader(self.filenames[file], inputdict={'--start_number':index, '-vframes':'3'})
            # reader = skvideo.io.vreader(self.filenames[file], inputdict={'-vf':'select=gte(n\\,{})'.format(index), '-vframes':'3'})
            triplet = []
            for frame in reader:
                triplet.append(frame)
        triplet = [torch.from_numpy(frame.transpose((2, 0, 1))).type('torch.FloatTensor') for frame in triplet] # (C, H, W)
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