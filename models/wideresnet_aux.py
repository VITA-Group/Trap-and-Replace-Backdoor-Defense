import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wideresnet import BasicBlock

class WideResNetAux(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, aux_drop_rate=0.0, stem_end_block=4):
        super(WideResNetAux, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = self._make_layer(n, channels[0], channels[1], 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = self._make_layer(n, channels[1], channels[2], 2, drop_rate)
        # 3rd block
        self.block3 = self._make_layer(n, channels[2], channels[3], 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        self.stem_end_block = stem_end_block
        if self.stem_end_block == 4:
            self.block3_aux = self._make_layer(n, channels[2], channels[3], 2, aux_drop_rate)
        elif self.stem_end_block == 5:
            self.block3_aux = self._make_layer(n, channels[2], channels[3], 2, aux_drop_rate)
            self.block3_aux[0] = None
        elif self.stem_end_block == 6:
            pass
        else:
            raise Exception('Unexpected stem_end_block: %d' % stem_end_block)
        self.bn1_aux = nn.BatchNorm2d(channels[3])
        self.fc_aux = nn.Linear(channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

        self.aux_drop_rate = aux_drop_rate

    def _make_layer(self, nb_layers, in_planes, out_planes, stride, drop_rate=0, activate_before_residual=False):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward_stem(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)

        if self.stem_end_block == 4:
            stem_out = out
        elif self.stem_end_block == 5:
            stem_out = self.block3[0](out)
        elif self.stem_end_block == 6:
            stem_out = self.block3(out)

        return stem_out

    def forward_main_branch(self, stem_out):
        # go main branch:
        if self.stem_end_block == 4:
            h3 = self.block3(stem_out)
        elif self.stem_end_block == 5:
            h3 = self.block3[1](stem_out)
        elif self.stem_end_block == 6:
            h3 = stem_out
        out = F.leaky_relu(self.bn1(h3), negative_slope=0.1, inplace=False)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        logits = self.fc(out)

        return logits

    def forward_aux_branch(self, stem_out):
        # go aux branch:
        if self.stem_end_block == 4:
            h3 = self.block3_aux(stem_out)
        elif self.stem_end_block == 5:
            h3 = self.block3_aux[1](stem_out)
        elif self.stem_end_block == 6:
            h3 = stem_out
        h3 = F.dropout(h3, p=self.aux_drop_rate, training=self.training)
        out = F.leaky_relu(self.bn1_aux(h3), negative_slope=0.1, inplace=False)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        logits_aux = self.fc_aux(out)

        return logits_aux

    def forward(self, x):
        
        stem_out = self.forward_stem(x)

        logits = self.forward_main_branch(stem_out)

        logits_aux = self.forward_aux_branch(stem_out)

        return logits, logits_aux



def WRN16AUX(num_classes=10, widen_factor=1, stem_end_block=4, aux_drop_rate=0.0):
    return WideResNetAux(depth=16, num_classes=num_classes, widen_factor=widen_factor, stem_end_block=stem_end_block, aux_drop_rate=aux_drop_rate)

if __name__ == '__main__':
    model = WRN16AUX(stem_end_block=4)
    for name, p in model.named_parameters():
        print(name, p.shape)

    # print(model.block3[0])
    # print(model.block3[0])

'''
conv1.weight torch.Size([16, 3, 3, 3])
conv1.bias torch.Size([16])
block1.0.bn1.weight torch.Size([16])
block1.0.bn1.bias torch.Size([16])
block1.0.conv1.weight torch.Size([16, 16, 3, 3])
block1.0.conv1.bias torch.Size([16])
block1.0.bn2.weight torch.Size([16])
block1.0.bn2.bias torch.Size([16])
block1.0.conv2.weight torch.Size([16, 16, 3, 3])
block1.0.conv2.bias torch.Size([16])
block1.1.bn1.weight torch.Size([16])
block1.1.bn1.bias torch.Size([16])
block1.1.conv1.weight torch.Size([16, 16, 3, 3])
block1.1.conv1.bias torch.Size([16])
block1.1.bn2.weight torch.Size([16])
block1.1.bn2.bias torch.Size([16])
block1.1.conv2.weight torch.Size([16, 16, 3, 3])
block1.1.conv2.bias torch.Size([16])
block2.0.bn1.weight torch.Size([16])
block2.0.bn1.bias torch.Size([16])
block2.0.conv1.weight torch.Size([32, 16, 3, 3])
block2.0.conv1.bias torch.Size([32])
block2.0.bn2.weight torch.Size([32])
block2.0.bn2.bias torch.Size([32])
block2.0.conv2.weight torch.Size([32, 32, 3, 3])
block2.0.conv2.bias torch.Size([32])
block2.0.convShortcut.weight torch.Size([32, 16, 1, 1])
block2.0.convShortcut.bias torch.Size([32])
block2.1.bn1.weight torch.Size([32])
block2.1.bn1.bias torch.Size([32])
block2.1.conv1.weight torch.Size([32, 32, 3, 3])
block2.1.conv1.bias torch.Size([32])
block2.1.bn2.weight torch.Size([32])
block2.1.bn2.bias torch.Size([32])
block2.1.conv2.weight torch.Size([32, 32, 3, 3])
block2.1.conv2.bias torch.Size([32])
block3.0.bn1.weight torch.Size([32])
block3.0.bn1.bias torch.Size([32])
block3.0.conv1.weight torch.Size([64, 32, 3, 3])
block3.0.conv1.bias torch.Size([64])
block3.0.bn2.weight torch.Size([64])
block3.0.bn2.bias torch.Size([64])
block3.0.conv2.weight torch.Size([64, 64, 3, 3])
block3.0.conv2.bias torch.Size([64])
block3.0.convShortcut.weight torch.Size([64, 32, 1, 1])
block3.0.convShortcut.bias torch.Size([64])
block3.1.bn1.weight torch.Size([64])
block3.1.bn1.bias torch.Size([64])
block3.1.conv1.weight torch.Size([64, 64, 3, 3])
block3.1.conv1.bias torch.Size([64])
block3.1.bn2.weight torch.Size([64])
block3.1.bn2.bias torch.Size([64])
block3.1.conv2.weight torch.Size([64, 64, 3, 3])
block3.1.conv2.bias torch.Size([64])
bn1.weight torch.Size([64])
bn1.bias torch.Size([64])
bn1_aux.weight torch.Size([64])
bn1_aux.bias torch.Size([64])
fc.weight torch.Size([10, 64])
fc.bias torch.Size([10])
fc_aux.weight torch.Size([10, 64])
fc_aux.bias torch.Size([10])
'''