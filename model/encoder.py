import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, Optional, List
import functools


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AttnBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        base_width: int = 64,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttnBottleneck(nn.Module):
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        base_width: int = 64,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        
        self.conv1 = conv1x1(inplanes, width)
        self.ln1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.ln2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.ln3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.ln3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FusionLayer(nn.Module):
    def __init__(self,
                 block: Type[Union[AttnBasicBlock, AttnBottleneck]],
                 layers: int,
                 input_channels: List[int] = [64, 128, 256],
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 width_per_group: int = 64,
                 ):
        super(FusionLayer, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
            
        self._norm_layer = norm_layer
        self.dilation = 1
        self.base_width = width_per_group
        
        conv_layers = []
        for input_channel in input_channels:
            conv_layers.append(self._make_conv_layer(block, input_channel * block.expansion, input_channels[-1]))
        self.conv_layers = nn.ModuleList(conv_layers)
        
        self.encode_layer1 = self._make_layer(block, input_channels[-1] * block.expansion * len(input_channels), input_channels[-1] * 2, layers, stride=2)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_conv_layer(self, block, inplanes: int, out_planes: int) -> nn.Sequential:
        layers = []
        norm_layer = self._norm_layer
        while out_planes * block.expansion != inplanes:
            layers.append(
                nn.Sequential(conv3x3(in_planes=inplanes, out_planes=inplanes * 2, stride=2),
                              norm_layer(inplanes * 2),
                              nn.ReLU(inplace=True))
            )
            inplanes = inplanes * 2
            
        return nn.Sequential(*layers)

    def _make_layer(self, block: Type[Union[AttnBasicBlock, AttnBottleneck]], inplanes: int, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer, base_width=self.base_width))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
       
        feature = [self.conv_layers[i](xi) for i, xi in enumerate(x)]
        feature = torch.cat(feature, dim=1)
        output = self.encode_layer1(feature)

        return output.contiguous()


class Encoder(nn.Module):
    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256], attn_block_num=3) -> None:
        super(Encoder, self).__init__()
        self.expansion = 4
        if backbone == 'resnet18':
            self.fusion_layer = FusionLayer(AttnBasicBlock, 2, input_channels)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.fusion_layer = FusionLayer(AttnBasicBlock, attn_block_num, input_channels)
            self.expansion = 1
            
        elif backbone == 'resnet50':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels)
        elif backbone == 'resnet101':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels)
        elif backbone == 'resnet152':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels)
        elif backbone == 'wide_resnet50_2':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2)
        elif backbone == 'wide_resnet101_2':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2)
            
    def forward(self, x):
        return self.fusion_layer(x)
