import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
sys.path.append("./src")
sys.path.append("./models")
from models.GAN_model import GANGenerator, GANDiscriminator

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
    discriminator = GANDiscriminator()
    generator = GANGenerator()
    D_optimizer = optim.Adam(discriminator.parameters(),lr=0.0002)
    G_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for index, sample in enumerate(dataloader):
            #inframes  (N,C,H,W,2), outframes (N,C,H,W)
            left, right, outframes = sample['left'], sample['right'], sample['out']
            inframes = (left, right)

            #train discriminator
            generated_data = generator(inframes).detach()
            D_loss, real_pred, generated_pred = train_D(discriminator,
                                                          D_optimizer,
                                                          outframes,
                                                          generated_data,
                                                          criterion)

            #train generator
            generated_data = generator(inframes)
            G_loss = train_G(discriminator, G_optimizer, generated_data, criterion)

            if index % 100 == 0:
                N = generated_data.shape[0]
                n_imgs = generated_data.data
                imshow(torchvision.utils.make_grid(n_imgs))
                print("epoch {} out of {}".format(epoch,epochs))
                print("D_loss:{}, G_loss:{}\n".format(D_loss,G_loss))
    return generator, discriminator


def train_D(discriminator,optimizer,real_data,generated_data,criterion):
    """
    :param discriminator: discriminator model
    :param optimizer: optimizer object
    :param real_data: data from dataset (N,C,H,W)
    :param generated_data: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :return:real_loss+gen_loss,real_output,generated_output
    """
    optimizer.zero_grad()
    N = real_data.shape[0]

    real_output = discriminator(real_data)
    real_loss = criterion(real_output,torch.ones(N,1))
    real_loss.backward()

    generated_output = discriminator(generated_data)
    gen_loss = criterion(generated_output,torch.zeros(N,1))
    gen_loss.backward()

    optimizer.step()

    return real_loss+gen_loss,real_output,generated_output

def train_G(discriminator,optimizer,generated_data,criterion):
    """
    :param discriminator: generator model
    :param optimizer: optimizer object
    :param generated_data: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :return: generated loss
    """
    N = generated_data.shape[0]

    optimizer.zero_grad()

    output = discriminator(generated_data)
    generated_loss = criterion(output,torch.ones(N,1))
    generated_loss.backward()

    optimizer.step()

    return generated_loss
