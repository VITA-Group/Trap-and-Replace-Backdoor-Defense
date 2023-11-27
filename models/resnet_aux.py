import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
        drop_rate: float = 0.
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.drop_rate = drop_rate

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetAE(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        stem_end_block: int = 31, 
        aux_drop_rate: float = 0.5, 
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetAE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.layer4_aux = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], drop_rate=aux_drop_rate)
        self.fc_aux = nn.Linear(512 * block.expansion, num_classes)

        self.aux_drop_rate = aux_drop_rate

        self.stem_end_block = stem_end_block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, drop_rate: float = 0.) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward_stem(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.stem_end_block == 27:
            stem_out = x
        elif self.stem_end_block == 29:
            stem_out = self.layer4[0](x)
        elif self.stem_end_block == 31:
            x = self.layer4[0](x)
            stem_out = self.layer4[1](x)
        elif self.stem_end_block == 33:
            stem_out = self.layer4(x)

        return stem_out

    def forward_main_branch(self, stem_out: Tensor) -> Tensor:
        if self.stem_end_block == 27:
            x = self.layer4(stem_out)
        elif self.stem_end_block == 29:
            x = self.layer4[1](stem_out)
            x = self.layer4[2](x)
        elif self.stem_end_block == 31:
            x = self.layer4[2](stem_out)
        elif self.stem_end_block == 33:
            x = stem_out

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_aux_branch(self, stem_out: Tensor) -> Tensor:

        if self.stem_end_block == 27:
            x = self.layer4_aux(stem_out)
        elif self.stem_end_block == 29:
            x = self.layer4_aux[1](stem_out)
            x = self.layer4_aux[2](x)
        elif self.stem_end_block == 31:
            x = self.layer4_aux[2](stem_out)
        elif self.stem_end_block == 33:
            x = stem_out
        
        x = F.dropout(x, p=self.aux_drop_rate, training=self.training)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_aux(x)

        return x

    def forward(self, x: Tensor) -> Type[Tuple[Tensor, Tensor]]:
        stem_out = self.forward_stem(x)
        logits = self.forward_main_branch(stem_out)
        logits_aux = self.forward_aux_branch(stem_out)
        return logits, logits_aux

def ResNet34AUX(num_classes=1000, stem_end_block=31, aux_drop_rate=0.5):
    return ResNetAE(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, stem_end_block=stem_end_block, aux_drop_rate=aux_drop_rate)

if __name__ == '__main__':
    model = ResNet34AUX(stem_end_block=31)
    # for name, p in model.named_parameters():
    #     print(name, p.shape)

    imgs = torch.zeros((2,3,224,224))
    logits, logits_aux = model(imgs)

    print(logits.shape)
    print(logits_aux.shape)