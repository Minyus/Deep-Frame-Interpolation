import torch
import torch.nn as nn
import numpy as np


class GAN_interpolation_Generator(torch.nn.Module):  # input is N X C X H X W x 2
    def __init__(self, conv_layers_size):
        '''
        :param N: Number of Tensors in minibatch
        :param conv_layers_size: how many conv layers to add in between
        '''
        super(GAN_interpolation_Generator, self).__init__()
        self.conv_first_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_first_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_final_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_first_list = nn.ModuleList(
            [nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2) for i in range(conv_layers_size - 1)])
        self.conv_final_list = nn.ModuleList(
            [nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2) for i in range(conv_layers_size - 1)])
        self.conv_first_last = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.conv_last_last = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.final_layer = torch.nn.Bilinear(1, 1, 1)  # simple bilinear layer to combine first and last frame
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        print(x.shape)
        x_first = self.conv_first_1(x[:,:, :, :,0])
        x_first = self.activation(x_first)

        x_final = self.conv_final_1(x[:,:, :, :, 1])
        x_final = self.activation(x_final)
        for i in range(len(self.conv_first_list)):
            x_first = self.conv_first_list[i](x_first)
            x_first = self.activation(x_first)

            x_final = self.conv_final_list[i](x_final)
            x_final = self.activation(x_final)

        x_first = self.conv_first_last(x_first)
        x_final = self.conv_last_last(x_final)
        output = self.final_layer(torch.unsqueeze(x_first,dim=-1),torch.unsqueeze(x_final,dim=-1))
        #output (N X C X H X W )
        return torch.squeeze(output.dim=-1)


class GAN_interpolation_Discriminator(torch.nn.Module):  # input is N X C X H X W
    def __init__(self,N, flattened_img_size, hidden_size):
        '''
        :param N: Number of Tensors in minibatch
        :param flattened_img_size: size of img when flattened
        :param hidden_size: size of hidden layer
        '''
        super(GAN_interpolation_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(N*16*flattened_img_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.activation = nn.LeakyReLU()
        self.final_activation = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.linear1(x.view(-1))
        x = self.linear2(x)
        return self.final_activation(x)
