import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("./src")
sys.path.append("./models")

def imsave(img,name="Overfit_test",path="./experiments/"):
    npimg = img.numpy().astype(np.uint8).transpose((1,2,0))
    #format H,W,C
    plt.figure(figsize=(20,10))
    plt.imshow(npimg)
    plt.savefig(path+name)


def evalGAN(dataloader,pathToModel):
    """
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
    with torch.no_grad():
        for index, sample in enumerate(dataloader):
            # print(index)
            #inframes  (N,C,H,W,2), outframes (N,C,H,W)
            left, right, outframes = sample['left'].type(dtype),sample['right'].type(dtype),sample['out'].type(dtype)
            inframes = (left, right)

            generated_data = generator(inframes)

            G_eval = nn.functional.mse_loss(generated_data,outframes)

            # G_loss = train_GS(discriminator,G_optimizer,outframes,generated_data,criterion,dtype,epoch)
            if index % 100 == 0:
                # N = generated_data.shape[0]
                n_imgs = generated_data.data.cpu()
                imsave(torchvision.utils.make_grid(n_imgs),name="Overfit_test_generated.png")
                imsave(torchvision.utils.make_grid(outframes.data.cpu()),name="Overfit_test_real.png")
                # print("mean red:{}, mean green:{},mean blue:{} ".format(n_imgs[:,0,:,:].mean(),
                #                                                         n_imgs[:,1,:,:].mean(),
                #                                                         n_imgs[:,2,:,:].mean()))
                print( "G_eval:{}".format(G_eval))