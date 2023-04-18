import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.PReLU):
            pass
        else:
            m.initialize()

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.layer1 = nn.Sequential(*feats[:13])
        self.layer2 = nn.Sequential(*feats[13:23])
        self.layer3 = nn.Sequential(*feats[23:33])
        self.layer4 = nn.Sequential(*feats[33:43])

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        
        return x1, x2, x3, x4, x5

    def initialize(self):
        vgg16 = models.vgg16_bn(pretrained=True)
        self.load_state_dict(vgg16.state_dict(), False)

class SplitConvBlock(nn.Module):
    def __init__(self, scales, channel=64):
        super(SplitConvBlock, self).__init__()
        assert scales >= 2 or scales <= 6, 'scales should be between 2 to 6'
        self.scales = scales
        self.width = math.ceil(channel/scales)
        if scales == 6:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.width,   self.width, 3, stride=1, padding=1, dilation=1, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.width+1, self.width, 3, stride=1, padding=1, dilation=1, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )
        if scales > 2:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.width+1, self.width, 3, stride=1, padding=2, dilation=2, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )
        if scales > 3:
            self.conv3 = nn.Sequential(
                nn.Conv2d(self.width+1, self.width, 3, stride=1, padding=4, dilation=4, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )
        if scales > 4:
            self.conv4 = nn.Sequential(
                nn.Conv2d(self.width+1, self.width, 3, stride=1, padding=6, dilation=6, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )
        if scales > 5:
            self.conv5 = nn.Sequential(
                nn.Conv2d(self.width+1, self.width, 3, stride=1, padding=8, dilation=8, bias=False), nn.BatchNorm2d(self.width), nn.PReLU()
            )

    def forward(self, x, y):
        spx = torch.split(x, self.width, 1) 
        if y is None:
            sp1 = self.conv1(spx[0])
        else:
            sp1 = self.conv1(torch.cat((spx[0], y), 1))
        y, _ = torch.max(sp1, dim=1, keepdim=True)
        if self.scales > 2:
            sp2 = self.conv2(torch.cat((spx[1], y), 1))
            y, _ = torch.max(sp2, dim=1, keepdim=True)
        if self.scales > 3:
            sp3 = self.conv3(torch.cat((spx[2], y), 1))
            y, _ = torch.max(sp3, dim=1, keepdim=True)
        if self.scales > 4:
            sp4 = self.conv4(torch.cat((spx[3], y), 1))
            y, _ = torch.max(sp4, dim=1, keepdim=True)
        if self.scales > 5:
            sp5 = self.conv5(torch.cat((spx[4], y), 1))
            y, _ = torch.max(sp5, dim=1, keepdim=True)

        if   self.scales == 2:
            x = torch.cat((sp1, spx[1], y), 1)
        elif self.scales == 3:
            x = torch.cat((sp1, sp2, spx[2], y), 1)
        elif self.scales == 4:
            x = torch.cat((sp1, sp2, sp3, spx[3], y), 1)
        elif self.scales == 5:
            x = torch.cat((sp1, sp2, sp3, sp4, spx[4], y), 1)
        elif self.scales == 6:
            x = torch.cat((sp1, sp2, sp3, sp4, sp5, spx[5], y), 1)

        return x

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self, scales, in_channel, channel=64):
        super(Decoder, self).__init__()
        self.convert = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, bias=False), nn.BatchNorm2d(channel), nn.PReLU(),
        )
        self.scb = SplitConvBlock(scales, channel)
        self.convs = nn.Sequential(
            nn.Conv2d(channel+1, channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(channel), nn.PReLU(),
            nn.Conv2d(channel,   channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(channel), nn.PReLU(),
            nn.Conv2d(channel, 1, 3, padding=1)
        )

    def forward(self, x, y):
        x = self.convert(x)
        x = self.scb(x, y)
        if y is None:
            y = self.convs(x)
        else:
            y = self.convs(x) + y

        return y

    def initialize(self):
        weight_init(self)

class SGNet_vgg16(nn.Module):
    def __init__(self, cfg):
        super(SGNet_vgg16, self).__init__()
        self.cfg      = cfg
        self.bkbone   = VGGNet()
        self.decoder1 = Decoder(2, 128)
        self.decoder2 = Decoder(3, 256)
        self.decoder3 = Decoder(4, 512)
        self.decoder4 = Decoder(5, 512)
        self.decoder5 = Decoder(6, 512)

        self.initialize()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.bkbone(x)
        x_size = x.size()[2:] 

        y5     = self.decoder5(x5, None)
        p5     = F.interpolate(y5, x_size, mode='bilinear', align_corners=True)

        y5_4   = F.interpolate(y5, x4.size()[2:], mode='bilinear', align_corners=True)
        y4     = self.decoder4(x4, y5_4)
        p4     = F.interpolate(y4, x_size, mode='bilinear', align_corners=True)

        y4_3   = F.interpolate(y4, x3.size()[2:], mode='bilinear', align_corners=True)
        y3     = self.decoder3(x3, y4_3)
        p3     = F.interpolate(y3, x_size, mode='bilinear', align_corners=True)

        y3_2   = F.interpolate(y3, x2.size()[2:], mode='bilinear', align_corners=True)	
        y2     = self.decoder2(x2, y3_2)
        p2     = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)

        y2_1   = F.interpolate(y2, x1.size()[2:], mode='bilinear', align_corners=True)	
        y1     = self.decoder1(x1, y2_1)
        p1     = F.interpolate(y1, x_size, mode='bilinear', align_corners=True)

        return p1, p2, p3, p4, p5

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
