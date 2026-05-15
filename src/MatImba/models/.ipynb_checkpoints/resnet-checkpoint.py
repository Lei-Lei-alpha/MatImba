import os
import time
import torch
import numpy as np
import torch.nn as nn
from ..models.fds import FDS
from ..utils.losses import *
from torch import Tensor, stack, optim, load, save
from ..utils.utils import AverageMeter, ProgressMeter, save_checkpoint

def swish(x, alpha = 0.08, beta = 0.9):
        return alpha * x * torch.sigmoid(x / beta)
    
def select_atv_func(name):
    atv_funcs = {
        'swish': swish,
        'silu': nn.SiLU(),
        'hardwish': nn.Hardswish(),
        'gelu': nn.GELU(),
        'celu': nn.CELU(alpha = 1),
        'mish': nn.Mish(),
        'tanhshrink': nn.Tanhshrink()
    }
    return atv_funcs[name]

act_func = select_atv_func('mish')

def convdata(in_channel, out_channel, stride = 1):
    """convolution with padding"""
    return nn.Conv1d(in_channel, out_channel, kernel_size = 1, stride = stride, padding = 0, bias = False)

class BasicBlock(nn.Module):
    expansion = 3
    def __init__(self, inplanes, planes, stride = 2, downsample = None):
        super().__init__()
        self.conv1 = convdata(inplanes, planes * 3, stride)
        self.bn1 = nn.BatchNorm1d(planes * 3)
        self.act_func = act_func
        self.conv2 = convdata(planes * 3, planes * 3)
        self.bn2 = nn.BatchNorm1d(planes*3)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual.float()
        out = self.act_func(out)
        return out

class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 7, downsample = None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes * 1, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm1d(planes * 1)
        self.conv2 = nn.Conv1d(planes, planes * 1, kernel_size = 1, stride = stride, padding = 0, bias=False)
        self.bn2 = nn.BatchNorm1d(planes * 1)
        # self.conv3 = nn.Conv1d(planes, planes * 1, kernel_size=1, bias = False)
        # self.bn3 = nn.BatchNorm1d(planes * 1)
        self.act_func = act_func
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.swish(out)
        # out = self.conv3(out)
        # out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual.float()
        out = self.act_func(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_size = None, output_size = None, block = Bottleneck,
                 dropout = None, ini_inplanes = 120, layers = [120, 256, 256],
                 load_from = None, fds = False, **fds_params):

        super().__init__()
        if load_from:
            self.checkpoint = torch.load(load_from)
            for key in self.checkpoint['init_params']:
                setattr(self, key, self.checkpoint['init_params'][key])
            self.inplanes = self.ini_inplanes
            if self.fds:
                self.FDS = FDS(**self.checkpoint['fds_params'])                
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.layers = layers
            self.block = block
            if dropout:
                self.dropout = nn.Dropout(p = dropout) 
            else:
                self.dropout = None
            self.ini_inplanes = ini_inplanes
            self.inplanes = self.ini_inplanes
            self.fds = fds
            try:
                self.fds_params = {**fds_params}
            except:
                pass
            
            if self.fds:
                self.fds_params.update({'feature_dim': self.layers[2] * self.block.expansion})
                self.FDS = FDS(**self.fds_params)
                
        self.conv1 = nn.Conv1d(self.input_size, self.layers[0], kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm1d(self.layers[0])
        self.act_func = act_func
        self.maxpool = nn.MaxPool1d(kernel_size = 1, stride = 5, padding = 0)
        self.layer1 = self._make_layer(self.layers[0], 1)
        self.layer2 = self._make_layer(self.layers[1], 1, stride = 7)
        # self.layer3 = self._make_layer(self.layers[2], 1, stride = 7)
        # self.layer4 = self._make_layer(256, 2, stride = 7)
        self.avgpool = nn.AvgPool1d(1, stride = 7)
        self.linear = nn.Linear(self.layers[2] * self.block.expansion, self.output_size)
        self.total_params = sum(p.numel() for p in self.parameters())
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]* m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        if load_from:
            self.load_state_dict(self.checkpoint['state_dict'])
                
    def _make_layer(self, planes, blocks = 1, stride = 5):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * self.block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm1d(planes * self.block.expansion),
            )
        layers = []
        layers.append(self.block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.block.expansion
        
        for i in range(1, blocks):
            layers.append(self.block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, targets = None, epoch = None):
        x = self.conv1(torch.unsqueeze(x.float(), -1))
        x = self.bn1(x)
        x = self.act_func(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)
        encoding_s = encoding.clone()

        if self.training and self.fds:
            if epoch is None:
                epoch = 0
            if epoch >= self.fds_params['start_smooth']:
                encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        if self.dropout:
            encoding_s = self.dropout(encoding_s)
        
        x = self.linear(encoding_s)

        if self.training and self.fds:
            return x, encoding
        else:
            return x
            