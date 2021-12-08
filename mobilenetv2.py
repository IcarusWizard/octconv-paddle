import h5py
import paddle
import paddle.nn as nn

from modules import *

class BottleneckV1(nn.Layer):
    """ResNetV1 BottleneckV1"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, alpha=0.5, zero_last_gamma=False):
        super().__init__()

        self.use_shortcut = stride == 1 and in_channels == out_channels

        # extract information
        self.conv1 = OctConv2D(in_channels, mid_channels, kernel_size=1, bias_attr=False, alpha_in=0, alpha_out=alpha)
        self.bn1 = OctBatchNorm(mid_channels, alpha)
        self.relu1 = Map(nn.ReLU6())

        # capture spatial relations
        self.conv2 = OctConv2D(mid_channels, mid_channels, kernel_size=3, padding=1, stride=stride, 
                               alpha_in=alpha, alpha_out=alpha, groups=mid_channels,  bias_attr=False)
        self.bn2 = OctBatchNorm(mid_channels, alpha)
        self.relu2 = Map(nn.ReLU6())

        # embeding back to information highway
        self.conv3 = OctConv2D(mid_channels, out_channels, kernel_size=1, bias_attr=False, alpha_in=alpha, alpha_out=0)
        self.bn3 = OctBatchNorm(out_channels, 0, weight_attr=nn.initializer.Constant(1.0) if (zero_last_gamma and self.use_shortcut) else None)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.use_shortcut:
            out = (safe_sum(out[0], x[0]), safe_sum(out[1], x[1]))

        return out


class MobileNetV2(nn.Layer):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, ratio=0., final_drop=0., zero_last_gamma=False):
        super().__init__()

        in_channels = [int(multiplier * x) for x in
                       [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        mid_channels = [int(t * x) for t, x in zip([1] + [6] * 16, in_channels)]
        out_channels = [int(multiplier * x) for t, x in zip([1] + [6] * 16,
                        [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320])]
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3
        in_ratios = [0.] + [ratio] * 13 + [0.] * 3
        ratios = [ratio] * 13 + [0.] * 4
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280

        self.conv1 = nn.Sequential(
            nn.Conv2D(3, int(32 * multiplier), kernel_size=3, padding=1, stride=2, bias_attr=False),
            nn.BatchNorm2D(int(32 * multiplier)),
            nn.ReLU6()
        )

        # ------------------------------------------------------------------
        stage_index, i = 1, 0
        for k, (in_c, mid_c, out_c, s, ir, r) in enumerate(zip(in_channels, mid_channels, out_channels, strides, in_ratios, ratios)):
            stage_index += 1 if s > 1 else 0
            i = 0 if s > 1 else (i + 1)
            name = 'L%d_B%d' % (stage_index, i)

            setattr(self, name, BottleneckV1(in_c, mid_c, out_c, stride=s, alpha=r, last_gamma=zero_last_gamma))

        # ------------------------------------------------------------------
        self.tail = nn.Sequential(
            nn.Conv2D(out_channels[-1], last_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(last_channels),
            nn.ReLU6()
        )

        # ------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0. else nn.Identity()
        self.classifer = nn.Conv2D(last_channels, classes, kernel_size=1)
        self.flat = nn.Flatten()

    def _get_channles(self, width, ratio):
        width = (width - int(ratio * width), int(ratio * width))
        width = tuple(c if c != 0 else -1 for c in width)
        return width

    def _concat(self, F, x1, x2):
        if (x1 is not None) and (x2 is not None):
            return F.Concat(x1, x2, dim=1)
        else:
            return x1 if x2 is None else x2

    def forward(self, x):

        x = self.conv1(x)

        x = (x, None)
        for iy in range(1, 10):
            # assume the max number of blocks is 50 per stage
            for ib in range(0, 50):
                name = 'L%d_B%d' % (iy, ib)
                if hasattr(self, name):
                    x = getattr(self, name)(x)

        x = self.tail(x[0])
        x = self.avgpool(x)
        x = self.drop(x)
        x = self.classifer(x)
        x = self.flat(x)
        return x
           

def mobilenet_v2_100(ratio=0.375, weight_file=None):
    net = MobileNetV2(multiplier=1.0, ratio=ratio)
    if weight_file is not None:
        if weight_file.endswith('.h5'):
            net = load_from_mxnet_weight(net, weight_file)
        else:
            net.load_dict(paddle.load(weight_file))
    return net
    
def mobilenet_v2_1125(ratio=0.5, weight_file=None):
    net = MobileNetV2(multiplier=1.125, ratio=ratio)
    if weight_file is not None:
        if weight_file.endswith('.h5'):
            net = load_from_mxnet_weight(net, weight_file)
        else:
            net.load_dict(paddle.load(weight_file))
    return net

def load_from_mxnet_weight(model, weight_file):
    named_params = dict(model.named_parameters())

    with h5py.File(weight_file, 'r') as f:
        for k in f.keys():
            v = f[k][:]
            k = k.replace('gamma', 'weight')
            k = k.replace('beta', 'bias')
            k = k.replace('running_mean', '_mean')
            k = k.replace('running_var', '_variance')
            named_params[k].set_value(paddle.to_tensor(v, dtype=paddle.float32))

    return model

if __name__ == '__main__':
    model = mobilenet_v2_1125(weight_file='weights/mobilenet_v2_1125_alpha-0.5.h5')