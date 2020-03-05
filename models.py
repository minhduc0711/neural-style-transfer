import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, vgg16


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dRefPad(num_filters, num_filters, 3)
        self.conv2 = Conv2dRefPad(num_filters, num_filters, 3)
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
    def __init__(self):
        super(TransformerNet, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(Conv2dRefPad(3, 32, 9),
                          nn.InstanceNorm2d(32, affine=True), nn.ReLU()),
            nn.Sequential(Conv2dRefPad(32, 64, 3, stride=2),
                          nn.InstanceNorm2d(64, affine=True), nn.ReLU()),
            nn.Sequential(Conv2dRefPad(64, 128, 3, stride=2),
                          nn.InstanceNorm2d(128, affine=True), nn.ReLU())
        ])
        self.residuals = nn.ModuleList([ResidualBlock(128)] * 5)
        self.convs_t = nn.ModuleList([
            nn.Sequential(UpsampleConv(128, 64, 3, stride=1, upsample=2),
                          nn.InstanceNorm2d(64, affine=True)),
            nn.Sequential(UpsampleConv(64, 32, 3, stride=1, upsample=2),
                          nn.InstanceNorm2d(32, affine=True))
        ])
        self.final_conv = Conv2dRefPad(32, 3, 9, stride=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            # print(x.shape)
        for residual in self.residuals:
            x = residual(x)
            # print(x.shape)
        for conv_t in self.convs_t:
            x = conv_t(x)
            # print(x.shape)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        # print(x.shape)
        return x
    
    
class Conv2dRefPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2dRefPad, self).__init__()
        pad = kernel_size // 2
        self.pad_layer = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        
    def forward(self, x):
        x = self.pad_layer(x)
        x = self.conv(x)
        return x

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=None):
        super(UpsampleConv, self).__init__()
        self.upsample = upsample
        pad = kernel_size // 2
        self.pad_layer = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
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