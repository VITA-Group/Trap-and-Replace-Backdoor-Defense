import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wideresnet import BasicBlock

class WideResNetAE(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, stem_end_block=4):
        super(WideResNetAE, self).__init__()
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
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
                nn.Sigmoid(),
            )
        elif self.stem_end_block == 5:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
                nn.Sigmoid(),
            )
        elif self.stem_end_block == 6:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
                nn.Sigmoid(),
            )
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
        rec = self.decoder(stem_out)

        return rec

    def forward(self, x):
        
        stem_out = self.forward_stem(x)

        logits = self.forward_main_branch(stem_out)

        rec = self.forward_aux_branch(stem_out)

        return logits, rec



def WRN16AE(num_classes=10, widen_factor=1, stem_end_block=4):
    return WideResNetAE(depth=16, num_classes=num_classes, widen_factor=widen_factor, stem_end_block=stem_end_block)

if __name__ == '__main__':
    model = WRN16AE(stem_end_block=6)
    for name, p in model.named_parameters():
        print(name, p.shape)

    imgs = torch.zeros((2,3,32,32))
    logits, imgs_rec = model(imgs)

    print(logits.shape)
    print(imgs_rec.shape)

