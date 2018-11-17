import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import skvideo.io
# import cv2
import numpy as np

class VideoInterpTripletsDataset(torch.utils.data.Dataset):
    def __init__(self,videoDirectory,tripletSize=256,aspectRatio=16.0/9.0,transform=None):
        """
        :param videoDirectory: Directory of videos
        :param tripletSize = pixel width of tripletDataset
        :param aspectRatio = aspect ratio of tripletDataset
        :param transform:
        """
        self.videoFilenames = [filename for filename in glob.glob(os.path.join(videoDirectory,'*.mp4'))]
        self.tripletSize = tripletSize
        self.aspectRatio = aspectRatio
        self.transform = transform
    def __len__(self):
        len = 0
        for filename in self.videoFilenames:
            len += int(skvideo.io.ffprobe(filename)['video']['@nb_frames'])
        return len
    def __getitem__(self,index):
        '''
        :param index: idx of the first frame in the video you want
        :return:
        '''
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

        reader = skvideo.io.vreader(correctFilename,inputdict={'--start_number':str(index),'-vframes':str(3)})
        H = self.tripletSize * self.aspectRatio
        W = self.tripletSize
        videoTriplet = []
        #each frame should be (H,W,C)
        for frame in reader:
            videoTriplet.append(frame)
        #stackedInterpFrames (H,W,C,2)
        stackedInterpFrames = np.stack([np.expand_dims(videoTriplet[0],axis=-1),
                                        np.expand_dims(videoTriplet[2],axis=-1)],axis=-1)
        #stackedInterpFrames (1,C,H,W,2)
        stackedInterpFrames = np.transpose(np.expand_dims(stackedInterpFrames,axis=0),(0,3,1,2,4))
        #missingFrame (1,C,H,W)
        missingFrame = np.transpose(np.expand_dims(videoTriplet[1],axis = 0),(0,3,1,2))
        # return a tuple of (stackedInterpFrames,missingFrame) #((1,C,H,W,2),(1,C,H,W))
        return stackedInterpFrames,missingFrame