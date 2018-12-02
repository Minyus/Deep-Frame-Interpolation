import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
sys.path.append("./src")
sys.path.append("./models")
import random
from models.GAN_model import GANGenerator, GANDiscriminator
from tqdm import tqdm_notebook

def imshow(img):
    print('Image dtype and mean')
    print(img.dtype)
    print(img.mean())
    output_image = img.numpy().transpose((1,2,0))
    npimg = np.interp(output_image,(-1.0,1.0),(0,255.0)).astype(np.uint8)
    print('Mean of image: {}'.format(npimg.mean()))
    #format H,W,C
    plt.imshow(npimg)
    plt.show()

def init_weights(m):
    for param in m.parameters():
        # print(param.shape)
        if len(param.shape) >= 2:
            # print("before: {}".format(param.data[0]))
            torch.nn.init.xavier_uniform_(param)
            # print("after: {}".format(param.data[0]))


def trainGAN(epochs, dataloader, save_path, save_every = None, supervised=True):
    """
    :param epochs: # of epochs to run for
    :param datasetloader: dataloader of dataset we want to train on
    :param save_path: path to where to save model
    :return: saved models
    """
    if save_every is None:
        save_every = epochs
    height, width = dataloader.dataset.getsize()
    print('Video (h,w): ({}, {})'.format(height,width))
    generator = GANGenerator(conv_layers_size=5)
    discriminator = GANDiscriminator(height=height, width=width, hidden_size=300)
    dtype = torch.FloatTensor
    print('Created models')

    if torch.cuda.is_available():
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        dtype = torch.cuda.FloatTensor
        print('GPU: {}'.format(torch.cuda.get_device_name(0)))

    discriminator.apply(init_weights)
    generator.apply(init_weights)
    print('Initialized weights')

    D_optimizer = optim.Adam(discriminator.parameters(),lr=0.0002)
    G_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    criterion = nn.BCELoss()
    print('Set up models')
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        index_for_sample = random.randint(0, len(dataloader))
        print('Index for sample: {}'.format(index_for_sample))
        with tqdm_notebook(total=len(dataloader)) as pbar:
            for index, sample in enumerate(dataloader):
                # print(index)
                #inframes  (N,C,H,W,2), outframes (N,C,H,W)
                left, right, outframes = sample['left'].type(dtype),sample['right'].type(dtype),sample['out'].type(dtype)
                inframes = (left, right)

                #train discriminator
                generated_data = generator(inframes).detach()
                D_loss, real_pred, generated_pred = train_D(discriminator,
                                                              D_optimizer,
                                                              outframes,
                                                              generated_data,
                                                              criterion,dtype)

                #train generator
                generated_data = generator(inframes)
                if not supervised:
                    G_loss = train_G(discriminator, G_optimizer, generated_data, criterion,dtype)
                else:
                    G_loss, G0_loss, S_loss = train_GS(discriminator,G_optimizer,outframes,generated_data,criterion,dtype,epoch)
                if index == index_for_sample:
                    N = generated_data.shape[0]
                    n_imgs = generated_data.data.cpu()
                    print('Generated images')
                    imshow(torchvision.utils.make_grid(n_imgs))
                    print('Real images')
                    imshow(torchvision.utils.make_grid(outframes.data.cpu()))
                    print("epoch {} out of {}".format(epoch,epochs))
                    print("D_loss:{}, G_loss:{}".format(D_loss,G_loss))
                    if supervised:
                        print("G_loss_only:{}, S_loss:{}".format(G0_loss, S_loss))
                    print("mean D_pred_real:{}, mean D_pred_gen:{}\n".format(real_pred.mean(),generated_pred.mean()))
                pbar.update(1)
        print('runtime: {}'.format(time.time() - start_time))
        if epoch % save_every == 0 and save_path is not None:
            torch.save(generator, '{}_{}_Generator'.format(save_path, epoch))
            torch.save(discriminator, '{}_{}_Discriminator'.format(save_path, epoch))
    if epochs % save_every != 0 and save_path is not None:
        torch.save(generator, '{}_{}_Generator'.format(save_path, epochs))
        torch.save(discriminator, '{}_{}_Discriminator'.format(save_path, epochs))

    return generator, discriminator


def train_D(discriminator,optimizer,real_data,generated_data,criterion,dtype):
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
    real_loss = criterion(real_output,torch.ones(N,1).type(dtype))
    real_loss.backward()

    generated_output = discriminator(generated_data)
    gen_loss = criterion(generated_output,torch.zeros(N,1).type(dtype))
    gen_loss.backward()

    optimizer.step()

    return real_loss+gen_loss,real_output,generated_output

def train_G(discriminator,optimizer,generated_data,criterion,dtype):
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
    generated_loss = criterion(output,torch.ones(N,1).type(dtype))
    generated_loss.backward()

    optimizer.step()

    return generated_loss
def train_GS(discriminator,optimizer,real_data,generated_data,criterion,dtype,epoch,lmd=0.1):
    """
    :param discriminator: generator model
    :param optimizer: optimizer object
    :param real_data: for supervised loss (N,C,H,W)
    :param generated_data: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :param epoch: adds to decay term of supervised loss
    :return: generated loss
    """
    N = generated_data.shape[0]

    optimizer.zero_grad()

    output = discriminator(generated_data)
    generated_loss = criterion(output,torch.ones(N,1).type(dtype))
    supervised_loss = torch.nn.functional.smooth_l1_loss(generated_data,real_data)
    # print("generated_loss: {}".format(generated_loss))
    total_loss = lmd * generated_loss + supervised_loss
    total_loss.backward()

    optimizer.step()
    return total_loss, generated_loss, supervised_loss
