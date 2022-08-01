"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1)
        nn.init.kaiming_normal_(conv_1.weight, nonlinearity="relu")

        conv_2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=(3, 3), padding=1)
        nn.init.kaiming_normal_(conv_2.weight, nonlinearity="relu")

        self.residual_function = nn.Sequential(
            conv_1,
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            conv_2,
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            nn.Dropout2d(0.1)
        )


        """input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)
        """

        """input_shape = in_channels
        residual_shape = out_channels * BasicBlock.expansion
        stride_width = int(round(input_shape / residual_shape))
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]"""

        #shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            conv_3 = nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=(1, 1), stride=(stride, stride))
            nn.init.kaiming_normal_(conv_3.weight, nonlinearity="relu")
            self.shortcut = nn.Sequential(
                conv_3,
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        nn.init.kaiming_normal_(conv_1.weight, nonlinearity="relu")

        conv_2 = nn.Conv2d(out_channels, out_channels, stride=(stride, stride), kernel_size=(3, 3), padding=1)
        nn.init.kaiming_normal_(conv_2.weight, nonlinearity="relu")

        conv_3 = nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=(1, 1))
        nn.init.kaiming_normal_(conv_3.weight, nonlinearity="relu")

        self.residual_function = nn.Sequential(
            conv_1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_3,
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=(stride, stride), kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, activation="linear"):
        super().__init__()

        self.activation = None
        if activation == "tanh":
            self.activation = nn.Tanh()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( (3, 3), stride=(2, 2), padding=1)
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.activation is not None:
            output = self.activation(output)

        return output

def resnet18(num_classes, activation="linear"):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, activation)

def resnet34(num_classes, activation="linear"):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, activation)

def resnet50(num_classes, activation="linear"):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, activation)

def resnet101(num_classes, activation="linear"):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, activation)

def resnet152(num_classes, activation="linear"):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, activation)