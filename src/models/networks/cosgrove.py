"""
MIT License

Copyright (c) 2022 NAME REDACTED
Copyright (c) 2017 Christian Cosgrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spec_norm
from itertools import chain
import numpy as np

########################
# Discriminator layers #
########################

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_fn=None):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(inplace=False),
                norm_fn(self.conv1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                norm_fn(self.conv2),
                #nn.BatchNorm2d(out_channels)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(inplace=False),
                norm_fn(self.conv1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                norm_fn(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass.weight.data, np.sqrt(2))
            self.bypass = norm_fn(self.bypass)
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_fn=None):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            norm_fn(self.conv1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            norm_fn(self.conv2),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            norm_fn(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)
    
class DiscriminatorImage(nn.Module):
    def __init__(self,
                 nf,
                 n_classes,
                 z_dim,
                 input_nc=3):
        """
        """
        super(DiscriminatorImage, self).__init__()

        self.features = nn.Sequential(
            FirstResBlockDiscriminator(input_nc, nf,
                                       stride=2,
                                       norm_fn=spec_norm),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2,
                                  norm_fn=spec_norm),
            ResBlockDiscriminator(nf*2, nf*4,
                                  stride=2,
                                  norm_fn=spec_norm),
            ResBlockDiscriminator(nf*4, nf*8,
                                  stride=2,
                                  norm_fn=spec_norm),
            ResBlockDiscriminator(nf*8, nf*8,
                                  stride=1,
                                  norm_fn=spec_norm),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        self.phi = spec_norm(self.get_linear(nf*8, 1))
        self.psi = nn.Sequential(
            spec_norm(self.get_linear(nf*8, nf*8)),
            nn.ReLU()
        )
        self.embed = nn.Embedding(n_classes, nf*8)
        self.z_dim = z_dim

        self.pred = nn.Sequential(
            spec_norm(self.get_linear(nf*8, nf*8)),
            #nn.BatchNorm1d(nf*8),
            nn.ReLU(),
            spec_norm(self.get_linear(nf*8, nf*8)),
            #nn.BatchNorm1d(nf*8),
            nn.ReLU(),
            spec_norm(self.get_linear(nf*8, z_dim*2))
        )
        #self.pred = spec_norm(nn.Linear(nf*8, z_dim*2))

    def finetune_parameters(self, embed_only=True):
        if embed_only:
            return self.embed.parameters()
        else:
            return chain(
                self.phi.parameters(),
                self.psi.parameters(),
                self.embed.parameters(),
                self.pred.parameters()
            )
        
    def get_linear(self, n_in, n_out):
        layer = nn.Linear(n_in, n_out)
        nn.init.xavier_uniform(layer.weight.data, 1.)
        return layer

    def forward(self, x, y):
        h = self.flatten(self.pool(self.features(x)))
        # Extract real/fake features and
        # pred_z/r features
        hh = self.psi(h)
        if y is None:
            rf = self.phi(hh)
        else:
            y_proj = self.embed(y)
            rf = self.phi(hh) + torch.sum(y_proj*hh, dim=1, keepdim=True)
        ###########
        cls_ = self.pred(h)
        ###########
        return rf, (cls_[:, 0:self.z_dim], cls_[:, self.z_dim:])
        

####################
# Generator layers #
####################

class CBN2d(nn.Module):
    def __init__(self, y_dim, bn_f):
        super(CBN2d, self).__init__()
        self.bn = nn.BatchNorm2d(bn_f, affine=False)
        self.scale = nn.Embedding(y_dim, bn_f)
        nn.init.xavier_uniform(self.scale.weight.data, 1.0)
        self.shift = nn.Embedding(y_dim, bn_f)
        nn.init.xavier_uniform(self.shift.weight.data, 0.)
        # https://github.com/pfnet-research/sngan_projection/blob/13c212a7f751c8f0cfd24bc5f35410a61ecb9a45/source/links/categorical_conditional_batch_normalization.py
        # Basically, they initialise with all ones for scale and all zeros for shift.
        # Though that is basically for a one-hot encoding, and we dont have that.
    def forward(self, x, y):
        scale = self.scale(y)
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)
        shift = self.shift(y)
        shift = shift.view(shift.size(0), shift.size(1), 1, 1)
        return self.bn(x)*scale + shift

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, y_dim, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.y_dim = y_dim

        self.bn = CBN2d(y_dim, in_channels)
        
        self.relu = nn.ReLU()
        self.ups = nn.Upsample(scale_factor=stride)

        self.bn2 = CBN2d(y_dim, out_channels)
        
        bypass = []
        if stride != 1:
            bypass.append(nn.Upsample(scale_factor=stride))
            if in_channels != out_channels:
                bypass.append(nn.Conv2d(in_channels, out_channels, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp, y):
        x = self.bn(inp, y)
        x = self.relu(x)
        x = self.ups(x)
        x = self.conv1(x)
        x = self.bn2(x, y)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)

class Generator(nn.Module):
    def __init__(self, n_channels, nf, n_classes, z_dim):
        super(Generator, self).__init__()
        self.nf = nf
        self.z_dim = z_dim
        
        self.dense = nn.Linear(self.z_dim, 4 * 4 * nf)
        self.final = nn.Conv2d(nf, n_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.rbn1 = ResBlockGenerator(nf, nf,
                                      n_classes,
                                      stride=2)
        self.rbn2 = ResBlockGenerator(nf, nf,
                                      n_classes,
                                      stride=2)
        self.rbn3 = ResBlockGenerator(nf, nf,
                                      n_classes,
                                      stride=2)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z, y):
        #return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        x = self.dense(z).view(-1, self.nf, 4, 4)
        x = self.rbn1(x, y)
        x = self.rbn2(x, y)
        x = self.rbn3(x, y)
        x = self.bn(x)
        x = self.relu(x)
        x = self.final(x)
        x = self.tanh(x)
        return x

def test_disc():
    disc = DiscriminatorImage(nf=64, input_nc=3, n_classes=100)
    xfake = torch.randn((4,3,32,32))
    yfake = torch.ones((4,1)).long()
    print(disc(xfake, yfake))
    
def test_gen():
    gen = Generator(n_channels=3, nf=64, n_classes=10, z_dim=128)
    z = torch.zeros((4,128)).normal_(0,1)
    y = torch.zeros((4,)).long()
    xfake = gen(z, y)
    print(xfake.shape)

if __name__ == '__main__':
    # test_disc()
    test_gen()