import torch
import torch.nn as nn

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


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class endeFUINT2_1(nn.Module):
    def __init__(self):
        super(endeFUINT2_1, self).__init__()
        dim = 64
        input_dim = 3
        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'

        self.conv0 = Conv2dBlock(3, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.down = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res2 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res3 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)

        self.res4 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res5 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res6 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim, 5, 1, 2,norm='none', activation=activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim, 5, 1, 2, norm='none', activation=activ, pad_type=pad_type)
        )

        self.out = Conv2dBlock(dim, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)

    def forward(self, x):
        x0 = self.conv0(x)
        x_d = self.down(x0)
        x1 = self.res1(x_d)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)

        x_u = self.up(x6)

        out = self.out(x_u)

        return out, [x1, x2, x3, x4, x5, x6]

class TeacherC2H(nn.Module):
    def __init__(self, input_dim=7):
        super(TeacherC2H, self).__init__()
        dim = 64

        activ = 'relu'
        pad_type = 'reflect'
        norm = 'none'

        self.conv0 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.down = nn.Sequential(
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type),
            Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        )

        self.res1 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res2 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res3 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res4 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res5 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)
        self.res6 = ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim, 5, 1, 2,norm='none', activation=activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim, 5, 1, 2, norm='none', activation=activ, pad_type=pad_type)
        )

        self.out = Conv2dBlock(dim, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)

    def forward(self, x):
        x0 = self.conv0(x)
        x_d = self.down(x0)
        x1 = self.res1(x_d)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)

        x_u = self.up(x6)
        out = self.out(x_u)

        return out, [x1, x2, x3, x4, x5, x6]
