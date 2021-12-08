""" 

Building Blocks for Octave Convolution

Note: These blocks are cases that octave level is 1 (i.e. you have 2^1 ways of information).

"""

import paddle
import paddle.nn as nn

def split_channels(channels, alpha):
    return int(channels * (1 - alpha)), channels - int(channels * (1 - alpha))

def safe_sum(x1, x2):
    if (x1 is not None) and (x2 is not None):
        return x1 + x2
    else:
        return x1 if x2 is None else x2

class Adapter(nn.Layer):
    def __init__(self):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_h, x_l = x
        if x_l is not None:
            x_l = self.upsampling(x_l)
            x_h = paddle.concat([x_h, x_l], axis=1)
        return x_h

class Map(nn.Layer):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        x_h, x_l = x
        x_h = self.func(x_h) if x_h is not None else None
        x_l = self.func(x_l) if x_l is not None else None
        return (x_h, x_l)

class OctBatchNorm(nn.Layer):
    def __init__(self, num_channels, alpha, weight_attr=None, bias_attr=None):
        super().__init__()
        # be compatible to conventional convolution
        c_h, c_l = split_channels(num_channels, alpha)

        self.bn_h = nn.BatchNorm2D(c_h, weight_attr=weight_attr, bias_attr=bias_attr) if c_h > 0 else nn.Identity()
        self.bn_l = nn.BatchNorm2D(c_l, weight_attr=weight_attr, bias_attr=bias_attr) if c_l > 0 else nn.Identity()

    def forward(self, x):
        x_h, x_l = x
        x_h = self.bn_h(x_h) if x_h is not None else None
        x_l = self.bn_l(x_l) if x_l is not None else None
        return (x_h, x_l)

class OctConv2D(nn.Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 alpha_in=0.5,
                 alpha_out=0.5,
                 enable_path=((1, 1), (1, 1)), 
                 groups=1, 
                 weight_attr=None,
                 bias_attr=None,
                 upsampling_mode='nearest'):
        super().__init__()

        # be compatible to conventional convolution
        (h2l, h2h), (l2l, l2h) = enable_path
        in_c_h, in_c_l = split_channels(in_channels, alpha_in)
        out_c_h, out_c_l = split_channels(out_channels, alpha_out) 

        # assert (in_c_h + in_c_l) == groups or ((in_c_h < 0 or in_c_h/groups >= 1) \
        #         and (in_c_l < 0 or in_c_l/groups >= 1)), \
        #     "Constains are not satisfied: (%d+%d)==%d, %d/%d>1, %d/%d>1" % ( \
        #     in_c_h, in_c_l, groups, in_c_h, groups, in_c_l, groups )
        # assert in_c_l != 0 and in_c_h != 0, \
        #     "TODO: current version has to specify the `in_channels' to determine the computation graph"
        # assert stride == 1 or stride == 2 or all((s <= 2 for s in stride)), \
        #     "TODO: current version only support stride({}) <= 2".format(stride)

        is_dw = False
        
        # computational graph will be automatic or manually defined
        self.enable_l2l = l2l and (in_c_l > 0 and out_c_l > 0)
        self.enable_l2h = l2h and (in_c_l > 0 and out_c_h > 0)
        self.enable_h2l = h2l and (in_c_h > 0 and out_c_l > 0)
        self.enable_h2h = h2h and (in_c_h > 0 and out_c_h > 0)

        if groups == (in_c_h + in_c_l): # depthwise convolution
            assert out_c_l == in_c_l and out_c_h == in_c_h
            self.enable_l2h, self.enable_h2l = False, False
            is_dw = True
        
        bias_attr_l2l, bias_attr_h2l = (False, bias_attr) if self.enable_h2l else (bias_attr, False)
        bias_attr_l2h, bias_attr_h2h = (False, bias_attr) if self.enable_h2h else (bias_attr, False)

        # deal with stride with resizing (here, implemented by pooling)
        s = (stride, stride) if type(stride) is int else stride
        do_stride2 = s[0] > 1 or s[1] > 1

        self.conv_l2l = None if not self.enable_l2l else nn.Conv2D(
                        in_c_l, out_c_l, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups if not is_dw else in_c_l,
                        weight_attr=weight_attr, bias_attr=bias_attr_l2l)

        self.conv_l2h = None if not self.enable_l2h else nn.Conv2D(
                        in_c_l, out_c_h, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups,
                        weight_attr=weight_attr, bias_attr=bias_attr_l2h)

        self.conv_h2l = None if not self.enable_h2l else nn.Conv2D(
                        in_c_h, out_c_l, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups,
                        weight_attr=weight_attr, bias_attr=bias_attr_h2l)

        self.conv_h2h = None if not self.enable_h2h else nn.Conv2D(
                        in_c_h, out_c_h, kernel_size=kernel_size, stride=1,
                        padding=padding, groups=groups if not is_dw else in_c_h,
                        weight_attr=weight_attr, bias_attr=bias_attr_h2h)

        self.l2l_down = nn.Identity() if not self.enable_l2l or not do_stride2 else \
                        nn.AvgPool2D(stride, ceil_mode=True)

        self.l2h_up = nn.Identity() if not self.enable_l2h or do_stride2 else \
                      nn.Upsample(scale_factor=2, mode=upsampling_mode)

        self.h2h_down = nn.Identity() if not self.enable_h2h or not do_stride2 else \
                        nn.AvgPool2D(stride, ceil_mode=True)

        self.h2l_down = nn.Identity() if not self.enable_h2l else \
                        nn.AvgPool2D(stride * 2, ceil_mode=True)

    def forward(self, x):
        x_high, x_low = x 

        x_h2h = self.conv_h2h(self.h2h_down(x_high)) if self.enable_h2h else None
        x_h2l = self.conv_h2l(self.h2l_down(x_high)) if self.enable_h2l else None

        x_l2h = self.l2h_up(self.conv_l2h(x_low)) if self.enable_l2h else None
        x_l2l = self.conv_l2l(self.l2l_down(x_low)) if self.enable_l2l else None

        x_h = safe_sum(x_l2h, x_h2h)
        x_l = safe_sum(x_l2l, x_h2l)
        return (x_h, x_l)
