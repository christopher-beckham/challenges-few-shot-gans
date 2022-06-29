import imp
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from functools import partial

from torch import optim
from torch.nn.utils import spectral_norm as spec_norm

from torchvision.models.resnet import resnet18

from itertools import chain

from .resnet_layers import ResidualBlock        

class MultipleResBlocks(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 z_dim,
                 n_classes,
                 n_blocks=1):
        super().__init__()


        # TODO support batch norm???

        rb0 = ResidualBlock(
            in_dim,
            out_dim,
            transposed_conv=True,
            with_coords=False,
            stride_input=True,
            with_film=True,
        )
        self.z0 = nn.Sequential(
            nn.Linear(z_dim*2, in_dim),
            nn.ReLU()
        )
        self.embed0 = nn.Embedding(n_classes, z_dim)
        self.rb0 = rb0
        
        rbs = []
        zs = []
        embeds = []
        for j in range(n_blocks):
            rbs.append(
                ResidualBlock(
                    out_dim,
                    out_dim,
                    transposed_conv=False,
                    with_coords=False,
                    stride_input=False,
                    with_film=True,
                )
            )
            zs.append(
                nn.Sequential(
                    nn.Linear(z_dim*2, in_dim),
                    nn.ReLU()
                )
            )
            embeds.append(nn.Embedding(n_classes, z_dim))

        self.rbs = nn.ModuleList(rbs)
        self.zs = nn.ModuleList(zs)
        self.embeds = nn.ModuleList(embeds)
        
    def forward(self, x, y, z):
        r = self.embed0(y)
        zr = torch.cat((z, r), dim=1)            
        zr_adain = self.z0(zr)
        x = self.rb0(x, zr_adain)
        for j in range(len(self.rbs)):
            r = self.embeds[j](y)
            zr = torch.cat((z,r), dim=1)
            zr_adain = self.zs[j](zr)
            x = self.rbs[j](x, zr_adain)
        return x
    
    def forward_mixup(self, x, y1, y2, alpha, z):
        assert alpha.ndim == 2
        r1 = self.embed0(y1)
        r2 = self.embed0(y2)
        r_mix = alpha*r1 + (1-alpha)*r2
        zr = torch.cat((z, r_mix), dim=1)
        zr_adain = self.z0(zr)
        x = self.rb0(x, zr_adain)
        for j in range(len(self.rbs)):
            r1 = self.embeds[j](y1)
            r2 = self.embeds[j](y2)
            r_mix = alpha*r1 + (1-alpha)*r2
            zr = torch.cat((z, r_mix), dim=1)
            zr_adain = self.zs[j](zr)
            x = self.rbs[j](x, zr_adain)
        return x

class GeneratorResNet(nn.Module):
    def __init__(self,
                 input_size,
                 input_nc,
                 z_dim,
                 n_classes,
                 blocks_per_res=[1,1,1,1],
                 ngf_decoder=None,
                 n_downsampling=3):
        
        super().__init__()
        
        assert n_downsampling == len(blocks_per_res)
                
        self.input_nc = input_nc
        self.input_size = input_size
        self.z_dim = z_dim

        h0_spatial_dim = int(input_size // (2**n_downsampling))
        self.h0 = nn.Parameter(
            torch.zeros((ngf_decoder,
                         h0_spatial_dim,
                         h0_spatial_dim)).normal_(0,1),
            requires_grad=True
        )
        print("self.h0: {}".format(self.h0.shape))
        
        blocks = []
        for i in range(n_downsampling):

            mult = 2**i
            mult2 = 2**(i+1)
            in_dim = ngf_decoder // mult
            out_dim = ngf_decoder // mult2
            
            blocks.append(MultipleResBlocks(
                in_dim=in_dim,
                out_dim=out_dim,
                z_dim=z_dim,
                n_classes=n_classes,
                n_blocks=blocks_per_res[i]
            ))
            
        self.dec_final = nn.Sequential(
            nn.Conv2d(
                ngf_decoder // (2**(i+1)),
                input_nc,
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()
        )        

        #self.encoder = nn.Sequential(*encoder)
        self.blocks = nn.ModuleList(blocks)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
    def finetune_parameters(self, embed_only=False):
        params = []
        for i in range(len(self.blocks)):
            params.append(self.blocks[i].embed0.parameters())
            params.append(self.blocks[i].embeds.parameters())
            if embed_only is False:
                params.append(self.blocks[i].z0.parameters())
                params.append(self.blocks[i].zs.parameters())
        return chain(*params)

    def forward(self, input):
        #enc = self.encoder(input)
        #dec = self.decoder(enc)
        #return dec
        raise NotImplementedError()

    def decode(self, y, z,  **kwargs):
        h = self.h0.repeat(z.size(0), 1, 1, 1)
        for j in range(len(self.blocks)):
            h = self.blocks[j](h, y, z)
        h = self.dec_final(h)
        return h

    def decode_mixup(self, y1, y2, alpha, z,  **kwargs):
        assert alpha.ndim == 2
        h = self.h0.repeat(z.size(0), 1, 1, 1)
        for j in range(len(self.blocks)):
            h = self.blocks[j].forward_mixup(h, y1, y2, alpha, z)
        h = self.dec_final(h)
        return h
    
if __name__ == '__main__':

    gen = GeneratorResNet(
        input_size=32,
        input_nc=1,
        output_nc=1,
        proj_dim=32,
        n_downsampling=4,
        ngf=32,
        ngf_decoder=128
    )
    print(gen)
    xfake = torch.randn((4,1,32,32))
    this_r = gen.encode(xfake)
    this_z = torch.zeros_like(this_r)

    x_fake = gen.decode(this_r, this_z)

    import pdb 
    pdb.set_trace()

#from .openai_vae import (Encoder, Decoder)
    