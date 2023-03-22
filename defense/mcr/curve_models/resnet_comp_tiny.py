import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from defense.mcr import curves


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=False)



class BasicBlockBase(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockBase, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # if self.downsample is not None:
        #     residual = self.downsample[0](x)
        #     residual = self.downsample[1](residual)
        out += self.downsample(x)

        out = F.relu(out)

        return out


class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, fix_points, stride=1):
        super(BasicBlockCurve, self).__init__()
        self.conv1 = curves.Conv2d(
            in_planes, planes, kernel_size=3, fix_points=fix_points, stride=stride, padding=1, bias=False
        )
        self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(
            planes, planes, kernel_size=3, fix_points=fix_points, stride=1, padding=1, bias=False
        )
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)

        self.downsample = nn.ModuleList()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.ModuleList(
                [curves.Conv2d(in_planes, self.expansion * planes, kernel_size=1, fix_points=fix_points, stride=stride, bias=False),
                curves.BatchNorm2d(self.expansion * planes, fix_points=fix_points)]
            )

    def forward(self, x, coeffs_t):
        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = F.relu(out)
        # out = F.relu(self.bn1(self.conv1(x, coeffs_t), coeffs_t))

        out = self.bn2(self.conv2(out, coeffs_t), coeffs_t)

        if len(self.downsample) > 0:
            residual = self.downsample[0](x, coeffs_t)
            residual = self.downsample[1](residual, coeffs_t)
            out += residual
        # out += self.downsample(x, coeffs_t)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)

        return out


class ResNetBase(nn.Module):
    def __init__(
        self, num_classes, block=BasicBlockBase, num_blocks=[2, 2, 2, 2], in_channel=3, zero_init_residual=False
    ):
        super(ResNetBase, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockBase):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNetCurve(nn.Module):
    def __init__(
        self, num_classes, fix_points, block=BasicBlockCurve, num_blocks=[2, 2, 2, 2], in_channel=3, zero_init_residual=False
    ):
        super(ResNetCurve, self).__init__()
        self.in_planes = 64

        self.conv1 = curves.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False, fix_points=fix_points
        )
        self.bn1 = curves.BatchNorm2d(64, fix_points=fix_points)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, fix_points=fix_points)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, fix_points=fix_points)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, fix_points=fix_points)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, fix_points=fix_points)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = curves.Linear(512 * block.expansion, num_classes, fix_points=fix_points)

        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.fill_(1)
                    getattr(m, 'bias_%d' % i).data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride, fix_points):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, fix_points=fix_points, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = F.relu(out)
        # out = F.relu(self.bn1(self.conv1(x, coeffs_t), coeffs_t))

        for block in self.layer1:
            out = block(out, coeffs_t)
        for block in self.layer2:
            out = block(out, coeffs_t)
        for block in self.layer3:
            out = block(out, coeffs_t)
        for block in self.layer4:
            out = block(out, coeffs_t)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out, coeffs_t)
        return out


class ResNet18:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'zero_init_residual': False}
# def resnet18(**kwargs):
#     backbone = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     #backbone.feature_dim = 512
#
#     return backbone
