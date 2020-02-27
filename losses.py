import torch, pdb
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from torchvision.models import vgg16


import numpy as np

class VGG_PerceptualLoss(_Loss):
    def __init__(self, vgg_layers=16):
        super(VGG_PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).cuda()
        self.loss_network = nn.Sequential(*list(vgg.features)[:vgg_layers]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        b, c, _, _ = input.size()
        if c == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if c != 1 and c != 3:
            raise("Wrong number of input channel: {}".format(c)) 
        return torch.mean(torch.pow(self.loss_network(input) - self.loss_network(target), 2))