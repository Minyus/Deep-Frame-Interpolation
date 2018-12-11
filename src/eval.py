import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
import numpy as np
from math import log10
import random
import os
sys.path.append("./src")
sys.path.append("./models")
from models.UNet_model import UNetGenerator, UNetDiscriminator
from models.GAN_model import GANGenerator, GANDiscriminator

from tqdm import tqdm_notebook

def imagesave(img,path="./experiments/"):
    output_image = img.numpy().transpose((1, 2, 0))
    npimg = np.interp(output_image, (-1.0, 1.0), (0, 255.0)).astype(np.uint8)
    #format H,W,C
    print(path)
    plt.figure(figsize=(20,10))
    plt.imshow(npimg)
    plt.savefig(path)


def evalGAN(dataloader,load_path,sampleImagesName = None,unet=True):
    """
    :param sampleImagesName: name of the
    :param dataloader: dataloader of eval dataset
    :param load_path: path to model
    :return:
    """
    assert(os.path.exists(load_path),"model dict does not exist")
    
    if unet:
        generator = UNetGenerator()
    else:
        generator = GANGenerator(conv_layers_size=5)
    generator.load_state_dict(torch.load(load_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    
    generator.eval()
    avg_psnr = 0
    index_for_sample = random.randint(0,len(dataloader))
    # print(index_for_sample)
    with torch.no_grad():
        with tqdm_notebook(total=len(dataloader)) as pbar:
            for index, sample in enumerate(dataloader):
                # print(index)
                #inframes  (N,C,H,W,2), outframes (N,C,H,W)
                left, right, outframes = sample['left'].to(device),sample['right'].to(device),sample['out'].to(device)
                inframes = (left, right)

                generated_data = generator(inframes)

                G_eval = nn.functional.mse_loss(generated_data,outframes)
                psnr = 10 * log10(1/G_eval.item())

                avg_psnr += psnr

                # print(index)
                # G_loss = train_GS(discriminator,G_optimizer,outframes,generated_data,criterion,dtype,epoch)
                if index == index_for_sample and sampleImagesName is not None:
                    # N = generated_data.shape[0]
                    n_imgs = generated_data.data.cpu()
                    imagesave(torchvision.utils.make_grid(n_imgs),path=sampleImagesName+"_generated.png")
                    imagesave(torchvision.utils.make_grid(outframes.data.cpu()),path=sampleImagesName+"_real.png")
                    # print("mean red:{}, mean green:{},mean blue:{} ".format(n_imgs[:,0,:,:].mean(),
                    #                                                         n_imgs[:,1,:,:].mean(),
                    #                                                         n_imgs[:,2,:,:].mean()))
                pbar.update(1)
#         print( "Avg. PNSR:{:.4f} dB".format(avg_psnr/len(dataloader)))
    return avg_psnr/len(dataloader)