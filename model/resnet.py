import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, downsample=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.downsample = downsample

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * self.expansion)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, downsample=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.downsample = downsample

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # print(out.size())
        # print(identity.size())
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=None, output_all_layer=False):
        super(ResNet, self).__init__()
        self.output_all_layer = output_all_layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], change_sample=False)
        self.layer2 = self._make_layer(block, 128, layers[1], change_sample=True)
        self.layer3 = self._make_layer(block, 256, layers[2], change_sample=True)
        self.layer4 = self._make_layer(block, 512, layers[3], change_sample=True)

    def _make_layer(self, block, planes, block_num, change_sample=False):
        """ 
        change_sample=True means the first block need change input size
        """
        downsample = None
        stride = 1
        if change_sample:
            stride = 2
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                self._norm_layer(planes * block.expansion))
        elif block is Bottleneck:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                self._norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, norm_layer=self._norm_layer, downsample=downsample))
        downsample = None
        stride = 1
        for _ in range(1, block_num):
            layers.append(block(planes * block.expansion, planes, stride=stride, norm_layer=self._norm_layer,
                                downsample=downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def init_weights(self, num_layers):
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        if self.output_all_layer:
            output = [layer1_out, layer2_out, layer3_out, layer4_out]
        else:
            output = layer4_out
        return output


# define net create function
def res_18(load_weight=False, output_all_layer=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_all_layer=output_all_layer)
    if load_weight:
        model.init_weights(18)
    return model


def res_34(load_weight=False, output_all_layer=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_all_layer=output_all_layer)
    if load_weight:
        model.init_weights(34)
    return model


def res_50(load_weight=False, output_all_layer=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_all_layer=output_all_layer)
    if load_weight:
        model.init_weights(50)
    return model


def res_101(load_weight=False, output_all_layer=False):
    model = ResNet(BasicBlock, [3, 4, 23, 3], output_all_layer=output_all_layer)
    if load_weight:
        model.init_weights(101)
    return model


def res_152(load_weight=False, output_all_layer=False):
    model = ResNet(BasicBlock, [3, 8, 36, 3], output_all_layer=output_all_layer)
    if load_weight:
        model.init_weights(152)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    import torch
    rs = res_34()
    summary(rs, (3, 512, 512), device='cpu')
    # inputs = torch.rand((1, 3, 256, 256))
    # res = rs(inputs)
    # print(res[0].size())  # 1/4
    # print(res[1].size())  # 1/8
    # print(res[2].size())  # 1/16
    # print(res[3].size())  # 1/32
