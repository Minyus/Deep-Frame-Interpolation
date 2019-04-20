# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:22:50 2019

@author: Monu
"""

#%load_ext autoreload
from src.utils import VideoInterpTripletsDataset
from src.train import trainGAN
from src.eval import evalGAN

from torch.utils.data import DataLoader
import torch
#%autoreload 2

dataset = VideoInterpTripletsDataset('datasets/frames/train', read_frames=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
valset = VideoInterpTripletsDataset('datasets/frames/val',read_frames=True)
valloader = DataLoader(valset,batch_size=32,shuffle=True,num_workers=4) 

#print(dataloader)
#print(valloader)

generator, discriminator = trainGAN(20, dataloader,valloader=valloader,supervised=True, save_path='./experiments/gan_supervised_only', save_every=1,gan=True)

#Runeval.py for eval
#valset = VideoInterpTripletsDataset('datasets/frames/val',read_frames=True)
#valloader = DataLoader(valset,batch_size=32,shuffle=True,num_workers=4)

#print(evalGAN(valloader,load_path = "./models/Model_SGAN",sampleImagesName="SGAN_val", unet=False))

#testset = VideoInterpTripletsDataset('datasets/frames/test',read_frames=True)
#testloader  = DataLoader(testset,batch_size=32,shuffle=True,num_workers=4)

#evalGAN(testloader,"./models/Model_SGAN",sampleImagesName="SGAN_test", unet=False)