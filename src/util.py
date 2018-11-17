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
        self.asepectRatio = aspectRatio
        self.transform = transform
    def __len__(self):
        len = 0
        for filename in self.videoFilenames:
            len += int(skvideo.io.ffprobe(filename)['video']['@nb_frames'])
        return len
    def __getitem__(self,videoFilename,idx):
        '''
        :param videoFilename: filename of the video you want
        :param idx: idx of the first frame in the video you want
        :return:
        '''
        if videoFilename in self.videoFilenames:
            reader = skvideo.io.vreader(videoFilename,inputdict={'--start_number':str(idx),'-vframes':str(3)})
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
            return (stackedInterpFrames,missingFrame)
        else:
            return AssertionError("videoFile: {} does not exist in the dataset".format(videoFilename))
        # tripletFilenames = [filename for filename in glob.glob(os.path.join(tripletDirectory,"*_triplets.mp4"))]
        # for filename in videoFilenames:
        #     if (filename[:-4] + '_triplets.mp4') in tripletFilenames:
        #         continue
        #     else:
        #         #this will be a (3,N,C,H,W) Tensor
        #         # N = # of frames / 3
        #         # (0,N,C,H,W) = frame 1's of triplet
        #         # (1,N,C,H,W) = frame 3's of triplet
        #         # (2,N,C,H,W) = frame 2's of triplet
        #         H = tripletSize * aspectRatio
        #         W = tripletSize
        #         videoTriplet = numpy.zeros_like([3,1,3,H,W])
        #         videoData = skvideo.io.vreader(filename)
        #         idx = 1
        #         #count which frame this is
        #         for frame in videoData:
        #             if idx % 2 == 0:
        #             else:
