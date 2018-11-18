import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
sys.path.append("./src")
sys.path.append("./models")
import GAN_model
import util

def imshow(img):
    npimg = img.numpy()
    #format H,W,C
    plt.imshow(npimg,(1,2,0))


def trainGAN(epochs,dataloader):
    """
    :param epochs: # of epochs to run for
    :param datasetloader: dataloader of dataset we want to train on
    :return: saved models
    """
    Discriminator = GAN_model.GAN_Discriminator()
    Generator = GAN_model.GAN_Generator()
    D_optimizer = optim.Adam(Discriminator.parameters(),lr=0.0002)
    G_optimizer = optim.Adam(Generator.parameters(), lr=0.0002)

    criterion = nn.BCELoss()

    for epoch in epochs:
        for index,sample in enumerate(dataloader):
            #inframes  (N,C,H,W,2), outframes (N,C,H,W)
            inframes,outframes = sample

            #train Discriminator
            generated_data = Generator(inframes).detach()
            D_loss,real_pred,generated_pred = train_D(Discriminator,
                                                          D_optimizer,
                                                          outframes,
                                                          generated_data,
                                                          criterion)


            #train Generator
            generated_data = Generator(inframes)
            G_loss = train_G(Discriminator,G_optimizer,generated_data,criterion)

            if index % 100 == 0:
                N = generated_data.shape[0]
                n_imgs = generated_data.data
                imshow(torchvision.utils.make_grid(n_imgs))
                print("epoch {} out of {}".format(epoch,epochs))
                print("D_loss:{}, G_loss:{}\n".format(D_loss,G_loss))
    return Generator,Discriminator




def train_D(Discriminator,optimizer,real_data,generated_data,criterion):
    """
    :param Discriminator: discriminator model
    :param optimizer: optimizer object
    :param real_data: data from dataset (N,C,H,W)
    :param generated_data: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :return:real_loss+gen_loss,real_output,generated_output
    """
    optimizer.zero_grad()
    N = real_data.shape[0]

    real_output = Discriminator(real_data)
    real_loss = criterion(real_output,torch.ones(N,1))
    real_loss.backward()

    generated_output = Discriminator(generated_data)
    gen_loss = criterion(generated_output,torch.zeros(N,1))
    gen_loss.backward()

    optimizer.step()

    return real_loss+gen_loss,real_output,generated_output

def train_G(Discriminator,optimizer,generated_data,criterion):
    """
    :param Discriminator: generator model
    :param optimizer: optimizer object
    :param generated_data: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :return: generated loss
    """
    N = generated_data.shape[0]

    optimizer.zero_grad()

    output = Discriminator(generated_data)
    generated_loss = criterion(output,torch.ones(N,1))
    generated_loss.backward()

    optimizer.step()

    return generated_loss
