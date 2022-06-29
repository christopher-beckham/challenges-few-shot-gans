"""
MIT License
Copyright (c) 2020 Thalles Silva
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

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models.resnet import resnet18

class Resnet18(nn.Module):
    def __init__(self, n_classes=10):
        super(Resnet18, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        h_dim = 512 # 512 for resnet18, 2048 for other classes
        self.f = nn.Sequential(*self.f,
                               nn.Flatten())
        self.cls = nn.Linear(h_dim, n_classes)

    def encode(self, x, **kwargs):
        return self.f(x)

    def forward(self, x):
        h = self.encode(x)
        pred = self.cls(h)
        return pred

if __name__ == '__main__':

    net = Resnet18(n_classes=10)

    print(net)