#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# Copyright 2020, NAME REDACTED
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform
from functools import partial

def coord_map(shape, start=-1, end=1):
  """
  Gives, a 2d shape tuple, returns two mxn coordinate maps,
  Ranging min-max in the x and y directions, respectively.
  """
  m, n = shape
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)


  coords = [x_coords, y_coords]
  
  for L in range(10):
    val = 2 ** L
    coords.append(torch.sin(val*np.pi*x_coords))
    coords.append(torch.cos(val*np.pi*x_coords))
    coords.append(torch.sin(val*np.pi*y_coords))
    coords.append(torch.cos(val*np.pi*y_coords))
    
  return torch.cat(coords, dim=0)



class Film(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, embedding):
    gammas = embedding[:, 0:(embedding.size(1)//2)]
    betas = embedding[:, (embedding.size(1)//2)::]
    if len(x.shape) == 4:
      gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
      betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    elif len(x.shape) == 5:
      gammas = gammas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
      betas = betas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
    else:
      raise Exception("")

    return (gammas * x) + betas

class ResidualBlock(nn.Module):
  def __init__(self, 
               in_dim, 
               out_dim=None,
               transposed_conv=False,
               with_residual=True,
               with_film=False,
               with_coords=False,
               norm_layer='instance',
               coord_repeat=1,
               coord_shape=None,
               stride_input=False):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock, self).__init__()

    with_batchnorm=True
    
    if norm_layer == 'instance':
        bn_class = partial(nn.InstanceNorm2d, affine=True)
    elif norm_layer == 'batch':
        bn_class = nn.BatchNorm2d
    else:
        raise Exception("norm_layer unknown")

    self.with_coords = with_coords
    coord_dim = 0
    if with_coords:
      if coord_shape is None:
        raise Exception("Need to specify spatial dim of coord layers")
      else:
        self.coords = torch.cat([coord_map(coord_shape)]*coord_repeat, dim=0)
        coord_dim = len(self.coords)

    if transposed_conv:
        conv_class2 = nn.ConvTranspose2d
        conv_kwargs2 = {'kernel_size': 3,
                        'padding': 1,
                        'output_padding': 1}
    else:
        conv_class2 = nn.Conv2d
        conv_kwargs2 = {'kernel_size': 3, 'padding': 1}
        
    conv_class1 = nn.Conv2d
    conv_kwargs1 =  {'kernel_size': 3, 'padding': 1}

    self.conv1 = conv_class1(in_dim+coord_dim,
                             out_dim,
                             **conv_kwargs1)
    self.conv2 = conv_class2(out_dim,
                             out_dim,
                             **conv_kwargs2,
                             stride=2 if stride_input else 1)
    self.with_batchnorm = with_batchnorm
    self.with_film = with_film

    if with_batchnorm:
      self.bn1 = bn_class(out_dim)
      self.bn2 = bn_class(out_dim)
    self.film = None
    if with_film:
      self.film = Film()
    self.with_residual = with_residual
    if not stride_input and (in_dim == out_dim or not with_residual):
      self.proj = None
    else:
      if stride_input:
        self.proj = nn.Sequential(
          conv_class1(in_dim+coord_dim,
                      out_dim, kernel_size=1),
          nn.Upsample(scale_factor=(2,2)) if transposed_conv \
              else nn.AvgPool2d(2)
        )
      else:
        self.proj = conv_class1(in_dim+coord_dim,
                                out_dim, kernel_size=1)

  def forward(self, x, embedding=None):
    orig_x = x
    if self.with_coords:
      coords = self.coords.repeat(x.size(0), 1, 1, 1)
      x = torch.cat((x, coords), dim=1)
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
      if self.film is not None:
        out = self.film(out, embedding)
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = orig_x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out

if __name__ == '__main__':
    layer = ResidualBlock(64, 64, 
                          with_coords=True,
                          coord_shape=(16,16),
                          stride_input=True)
    print(layer)