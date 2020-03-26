import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dRefPad(num_filters, num_filters, 3, bias=False)
        self.conv2 = Conv2dRefPad(num_filters, num_filters, 3, bias=False)
        self.inst_norm1 = nn.InstanceNorm2d(num_filters, affine=True)
        self.inst_norm2 = nn.InstanceNorm2d(num_filters, affine=True)

    def forward(self, x):
        old_x = x
        x = self.conv1(x)
        x = self.inst_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.inst_norm2(x)
        x += old_x

        return x


class TransformerNet(nn.Module):
    def __init__(self, alpha=1.0):
        """
        Build a generative network described in https://cs.stanford.edu/people/jcjohns/eccv16/
        but BatchNorm were replaced by InstanceNorm

        Args:
            alpha: factor to adjust the number of filters in conv layers

        """
        super(TransformerNet, self).__init__()

        # Number of filters for each conv layer
        conv_filters = [32] * 3  # [32, 64, 128]
        conv_ksizes = [9, 3, 3]
        conv_strides = [1, 2, 2]
        conv_filters = [int(n * alpha) for n in conv_filters]

        res_filters = [32] * 3  # default is 128
        res_filters = [int(n * alpha) for n in res_filters]

        upsample_filters = [32] * 2  # [64, 32]
        upsample_ksizes = [3, 3]
        upsample_strides = [1, 1]
        upsample_factors = [2, 2]
        upsample_filters = [int(n * alpha) for n in upsample_filters]

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()

        in_c = 3
        for out_c, ksize, stride in zip(conv_filters, conv_ksizes, conv_strides):
            self.convs.append(nn.Sequential(
                Conv2dRefPad(in_c, out_c, ksize, stride, bias=False),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU()
            ))
            in_c = out_c
        for out_c in res_filters:
            self.residuals.append(ResidualBlock(out_c))
            in_c = out_c
        for out_c, ksize, stride, factor in \
                zip(upsample_filters, upsample_ksizes, upsample_strides, upsample_factors):
            self.upsample_convs.append(nn.Sequential(
                UpsampleConv(in_c, out_c, ksize, stride,
                             upsample=factor, bias=False),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU(),
            ))
            in_c = out_c

        self.final_conv = Conv2dRefPad(int(32 * alpha), 3, 9, stride=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            # print(x.shape)
        for residual in self.residuals:
            x = residual(x)
            # print(x.shape)
        for upsample_conv in self.upsample_convs:
            x = upsample_conv(x)
            # print(x.shape)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        # print(x.shape)
        return x


class Conv2dRefPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(Conv2dRefPad, self).__init__()
        pad = kernel_size // 2
        self.pad_layer = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        x = self.pad_layer(x)
        x = self.conv(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=None, bias=True):
        super(UpsampleConv, self).__init__()
        self.upsample = upsample
        pad = kernel_size // 2
        self.pad_layer = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, bias=bias)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest',
                              scale_factor=float(self.upsample))
        x = self.pad_layer(x)
        x = self.conv(x)
        return x


class Vgg16Wrapper(nn.Module):
    def __init__(self, requires_grad):
        super(Vgg16Wrapper, self).__init__()
        features = list(vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x,
                content_layer_idxs=None,
                style_layer_idxs=None):
        if content_layer_idxs is None:
            content_layer_idxs = []
        if style_layer_idxs is None:
            style_layer_idxs = []

        content_acts = []
        style_acts = []
        max_idx = max(content_layer_idxs + style_layer_idxs)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in content_layer_idxs:
                content_acts.append(x)
            if i in style_layer_idxs:
                style_acts.append(x)
            if i >= max_idx:
                break
        return content_acts, style_acts
