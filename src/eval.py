import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
import numpy as np
from math import log10
import random
sys.path.append("./src")
sys.path.append("./models")

def imagesave(img,name="Overfit_test",path="./experiments/"):
    output_image = img.numpy().transpose((1, 2, 0))
    npimg = np.interp(output_image, (-1.0, 1.0), (0, 255.0)).astype(np.uint8)
    #format H,W,C
    print(path+name)
    plt.figure(figsize=(20,10))
    plt.imshow(npimg)
    plt.savefig(path+name)


def evalGAN(dataloader,pathToModel,sampleImagesName = None):
    """
    :param sampleImagesName: name of the
    :param dataloader: dataloader of eval dataset
    :param pathToModel: path to model Discrim and Gen
    :return:
    """
    generator = torch.load(pathToModel+"_Generator")
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        generator = generator.cuda()
        dtype = torch.cuda.FloatTensor

    generator.eval()
    avg_psnr = 0
    index_for_sample = random.randint(0,len(dataloader))

    # print(index_for_sample)
    with torch.no_grad():
        for index, sample in enumerate(dataloader):
            # print(index)
            #inframes  (N,C,H,W,2), outframes (N,C,H,W)
            left, right, outframes = sample['left'].type(dtype),sample['right'].type(dtype),sample['out'].type(dtype)
            inframes = (left, right)

            generated_data = generator(inframes)

            G_eval = nn.functional.mse_loss(generated_data,outframes)
            psnr = 10 * log10(1/G_eval.item())

            avg_psnr += psnr

            # print(index)
            # G_loss = train_GS(discriminator,G_optimizer,outframes,generated_data,criterion,dtype,epoch)
            if index == index_for_sample and sampleImagesName is not None:
                print("hit")
                # N = generated_data.shape[0]
                n_imgs = generated_data.data.cpu()
                imagesave(torchvision.utils.make_grid(n_imgs),name=sampleImagesName+"_generated.png")
                imagesave(torchvision.utils.make_grid(outframes.data.cpu()),name=sampleImagesName+"_real.png")
                # print("mean red:{}, mean green:{},mean blue:{} ".format(n_imgs[:,0,:,:].mean(),
                #                                                         n_imgs[:,1,:,:].mean(),
                #                                                         n_imgs[:,2,:,:].mean()))
        print( "Avg. PNSR:{:.4f} dB".format(avg_psnr/len(dataloader)))