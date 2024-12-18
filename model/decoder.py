from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, List, Optional
import functools

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class DeBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        base_width: int = 64
    ) -> None:
        super(DeBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        base_width: int = 64
    ) -> None:
        super(DeBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride)
        else:
            self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[DeBasicBlock, DeBottleneck]],
        layers: List[int],
        output_channels: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        width_per_group: int = 64
    ) -> None:
        super(DeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        self._norm_layer = norm_layer

        self.dilation = 1
        self.base_width = width_per_group
        deconv_layers = []
        
        for o_channel, layer in zip(output_channels[::-1], layers):
            deconv_layers.append(self._make_layer(block, o_channel * 2 * block.expansion, o_channel, layer, 2))
        self.layers = nn.ModuleList(deconv_layers)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block: Type[Union[DeBasicBlock, DeBottleneck]], inplanes: int, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, upsample, norm_layer, self.base_width))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            outputs.append(x)
        return outputs[::-1]


class Decoder(nn.Module):
    def __init__(self, backbone='wide_resnet50_2', output_channels=[64, 128, 256]):
        super(Decoder, self).__init__()
        if backbone == 'resnet18':
            self.backbone = DeResNet(DeBasicBlock, [2, 2, 2, 2], output_channels)
        elif backbone == 'resnet34':
            self.backbone = DeResNet(DeBasicBlock, [3, 4, 6, 3], output_channels)
        elif backbone == 'resnet50':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 6, 3], output_channels)
        elif backbone == 'resnet101':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 23, 3], output_channels)
        elif backbone == 'resnet152':
            self.backbone = DeResNet(DeBottleneck, [3, 8, 36, 3], output_channels)
        elif backbone == 'wide_resnet50_2':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 6, 3], output_channels, width_per_group=64 * 2)
        elif backbone == 'wide_resnet101_2':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 23, 3], output_channels, width_per_group=64 * 2)
            
    def forward(self, x):
        return self.backbone(x)
    