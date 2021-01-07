import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter as P
import numpy as np
import math
from torch.nn.modules.batchnorm import _BatchNorm


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y*x


class SEBasicBlockSW1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, with_norm=False):
        super(SEBasicBlockSW1, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes, 1)

        self.model1 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

        self.se = SELayer(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        residule = x
        out = self.conv1(x)
        out = self.relu(out)
        out0 = self.conv2(out)

        sw = self.model1(out0)

        out1 = sw * out0

        out = self.se(out1)
        out = residule + out

        return out

class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Student1(nn.Module):
    def __init__(self, intc=3,outc =3):
        super(Student1, self).__init__()
        dim = 64
        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'

        self.conv0 = nn.Sequential(
            Conv2dBlock(intc, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type),
            # Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.down1 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
            # Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.down2 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
            # Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res2 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res3 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res4 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res5 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res6 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.up1 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
        )

        self.up2 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
        )

        self.out = nn.Sequential(
            # Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
            Conv2dBlock(dim, outc, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type),
        )
        self.convR1 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR2 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR3 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x_d1 = self.down1(x0)
        x_d2 = self.down2(x_d1)

        x1 = x_d2 + self.res1(x_d2)
        x2 = x1 + self.res2(x1)
        # x21 = self.convR1(x2)

        x3 = x2 + self.res3(x2)
        x4 = x3 + self.res4(x3)
        # x41 = self.convR2(x4)

        x5 = x4 + self.res5(x4)
        x6 = x5 + self.res6(x5)
        # x61 = self.convR3(x6)

        x_u1 = self.up1(torch.cat([F.upsample_bilinear(x6, scale_factor=2), x_d1], 1))
        x_u2 = self.up2(torch.cat([F.upsample_bilinear(x_u1, scale_factor=2), x0], 1))
        out = self.out(x_u2)

        return out, [x1,x2,x3,x4,x5,x6]

class Student2(nn.Module):
    def __init__(self, intc=3,outc =3):
        super(Student2, self).__init__()
        dim = 64
        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'

        self.conv0 = nn.Sequential(
            Conv2dBlock(intc, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type),
            # Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.down1 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
            # Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.down2 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
            # Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res2 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res3 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res4 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res5 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res6 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.up1 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
        )

        self.up2 = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
        )

        self.out = nn.Sequential(
            # Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type),
            Conv2dBlock(dim, outc, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x_d1 = self.down1(x0)
        x_d2 = self.down2(x_d1)

        x1 = x_d2 + self.res1(x_d2)
        x2 = x1 + self.res2(x1)
        x3 = x2 + self.res3(x2)
        x4 = x3 + self.res4(x3)
        x5 = x4 + self.res5(x4)
        x6 = x5 + self.res6(x5)

        x_u1 = self.up1(torch.cat([F.upsample_bilinear(x6, scale_factor=2), x_d1], 1))
        x_u2 = self.up2(torch.cat([F.upsample_bilinear(x_u1, scale_factor=2), x0], 1))
        out = self.out(x_u2)

        return out, [x1, x2, x3, x4, x5, x6]
        # return out, [x2, x4, x6]

class StudentSmall(nn.Module):
    def __init__(self, intc=3, outc =3):
        super(StudentSmall, self).__init__()
        dim = 32
        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'

        self.conv0 = nn.Sequential(
            Conv2dBlock(intc, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type),
        )

        self.down1 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.down2 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res2 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res3 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res4 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res5 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res6 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.up1 = Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)
        self.up2 = Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)
        self.out = Conv2dBlock(dim, outc, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type)

        self.convR1 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR2 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR3 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR4 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR5 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR6 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)

        self.convR21 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR41 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.convR61 = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x_d1 = self.down1(x0)
        x_d2 = self.down2(x_d1)

        x1 = x_d2 + self.res1(x_d2)
        x2 = x1 + self.res2(x1)
        x3 = x2 + self.res3(x2)
        x4 = x3 + self.res4(x3)
        x5 = x4 + self.res5(x4)
        x6 = x5 + self.res6(x5)

        x_u1 = self.up1(torch.cat([F.upsample_bilinear(x6, scale_factor=2), x_d1], 1))
        x_u2 = self.up2(torch.cat([F.upsample_bilinear(x_u1, scale_factor=2), x0], 1))
        out = self.out(x_u2)

        feas = []
        feas.append(self.convR1(x1))
        feas.append(self.convR2(x2))
        feas.append(self.convR3(x3))
        feas.append(self.convR4(x4))
        feas.append(self.convR5(x5))
        feas.append(self.convR6(x6))

        return out, feas


class FakeTrans(nn.Module):
    def __init__(self, dim1, dim2):
        super(FakeTrans, self).__init__()
        self.conv1 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim2, dim1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        mimicF = self.conv1(f)

        fout =  F.relu(self.conv2(mimicF)) + f

        return fout, mimicF

class StudentSmallFollow(nn.Module):
    def __init__(self, intc=3, outc =3):
        super(StudentSmallFollow, self).__init__()
        dim = 32
        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'

        self.conv0 = nn.Sequential(
            Conv2dBlock(intc, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type),
        )

        self.down1 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.down2 = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res2 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res3 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res4 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res5 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.res6 = nn.Sequential(
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            SEBasicBlockSW1(dim, dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

        self.up1 = Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)
        self.up2 = Conv2dBlock(dim * 2, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)
        self.out = Conv2dBlock(dim, outc, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type)

        self.convR1 = FakeTrans(dim, 64)
        self.convR2 = FakeTrans(dim, 64)
        self.convR3 = FakeTrans(dim, 64)
        self.convR4 = FakeTrans(dim, 64)
        self.convR5 = FakeTrans(dim, 64)
        self.convR6 = FakeTrans(dim, 64)

    def forward(self, x):
        x0 = self.conv0(x)
        x_d1 = self.down1(x0)
        x_d2 = self.down2(x_d1)

        x1 = x_d2 + self.res1(x_d2)
        x1, x11 = self.convR1(x1)

        x2 = x1 + self.res2(x1)
        x2, x21 = self.convR2(x2)

        x3 = x2 + self.res3(x2)
        x3, x31 = self.convR3(x3)

        x4 = x3 + self.res4(x3)
        x4, x41 = self.convR4(x4)

        x5 = x4 + self.res5(x4)
        x5, x51 = self.convR5(x5)

        x6 = x5 + self.res6(x5)
        x6, x61 = self.convR6(x6)

        x_u1 = self.up1(torch.cat([F.upsample_bilinear(x6, scale_factor=2), x_d1], 1))
        x_u2 = self.up2(torch.cat([F.upsample_bilinear(x_u1, scale_factor=2), x0], 1))
        out = self.out(x_u2)

        return out, [x11, x21, x31, x41, x51, x61]
