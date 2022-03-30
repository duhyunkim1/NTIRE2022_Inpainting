import time
import datetime
import torch
import math
from torch import nn, optim
from torch.nn import functional as F
from modules import ConvLayer, ResBlk, DeformConv2d, ASPP
import numpy as np

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_rgb = ConvLayer(3, 32, kernel_size=3)
        self.to_rgb = ConvLayer(32, 3, kernel_size=3)

        # self.encode1 = nn.Sequential(*[nn.LeakyReLU(0.2), ConvLayer(32, 64, kernel_size=3, stride=2)])
        self.encode1 = nn.Sequential(*[ResBlk(32, 32), ASPP(32, 32), ResBlk(32, 64, downsample=True)])
        self.encode2 = nn.Sequential(*[ResBlk(64, 64, normalize=True), ASPP(64, 64), ResBlk(64, 128, normalize=True, downsample=True)])
        self.encode3 = nn.Sequential(*[ResBlk(128, 128, normalize=True), ASPP(128, 128), ResBlk(128, 256, normalize=True, downsample=True)])
        self.encode4 = nn.Sequential(*[ResBlk(256, 256, normalize=True), ASPP(256, 256), ResBlk(256, 512, normalize=True, downsample=True)])
        self.encode5 = nn.Sequential(*[ResBlk(512, 512, normalize=True), ASPP(512, 512), ResBlk(512, 512, normalize=True, downsample=True)])
        self.encode6 = nn.Sequential(*[ResBlk(512, 512, normalize=True), ASPP(512, 512), ResBlk(512, 512, normalize=True, downsample=True)])

        self.decode6 = nn.Sequential(*[ResBlk(512, 512, normalize=True), ASPP(512, 512), ResBlk(512, 512, normalize=True)])
        self.decode5 = nn.Sequential(*[ResBlk(512, 512, normalize=True), ASPP(512, 512), ResBlk(512, 512, normalize=True)])
        self.decode4 = nn.Sequential(*[ResBlk(512, 512, normalize=True), ASPP(512, 512), ResBlk(512, 256, normalize=True)])
        self.decode3 = nn.Sequential(*[ResBlk(256, 256, normalize=True), ASPP(256, 256), ResBlk(256, 128, normalize=True)])
        self.decode2 = nn.Sequential(*[ResBlk(128, 128, normalize=True), ASPP(128, 128), ResBlk(128, 64, normalize=True)])
        self.decode1 = nn.Sequential(*[ResBlk(64, 64), ASPP(64, 64), ResBlk(64, 32)])

        #super resolution ?????

    def forward(self, img):
        feat0 = self.from_rgb(img)
        feat1 = self.encode1(feat0)
        feat2 = self.encode2(feat1)
        feat3 = self.encode3(feat2)
        feat4 = self.encode4(feat3)
        feat5 = self.encode5(feat4)
        feat = self.encode6(feat5)

        feat = F.interpolate(self.decode6(feat), size=(feat5.size(2), feat5.size(3)), mode='bilinear') + feat5
        feat = F.interpolate(self.decode5(feat), size=(feat4.size(2), feat4.size(3)), mode='bilinear') + feat4
        feat = F.interpolate(self.decode4(feat), size=(feat3.size(2), feat3.size(3)), mode='bilinear') + feat3
        feat = F.interpolate(self.decode3(feat), size=(feat2.size(2), feat2.size(3)), mode='bilinear') + feat2
        feat = F.interpolate(self.decode2(feat), size=(feat1.size(2), feat1.size(3)), mode='bilinear') + feat1
        feat = F.interpolate(self.decode1(feat), size=(feat0.size(2), feat0.size(3)), mode='bilinear') + feat0

        # refinement network


        out = torch.sigmoid(self.to_rgb(feat))

        return out
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 64, kernel_size=3, stride=2)
        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = ConvLayer(256, 512, kernel_size=3, stride=2)

        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv1x1 = nn.Conv2d(512, 1, kernel_size=1, stride = 1, padding=0)

    def forward(self, x):        
        feat1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        feat2 = F.leaky_relu(self.bn2(self.conv2(feat1)), negative_slope=0.2, inplace=True)
        feat3 = F.leaky_relu(self.bn3(self.conv3(feat2)), negative_slope=0.2, inplace=True)
        feat4 = F.leaky_relu(self.bn4(self.conv4(feat3)), negative_slope=0.2, inplace=True)
        prob = self.conv1x1(feat4)
        return prob