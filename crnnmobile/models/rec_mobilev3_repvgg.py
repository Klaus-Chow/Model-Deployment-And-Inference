import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.insert(0,'./')
import torch
from models.repvggblock import RepVGGBlock,repvgg_model_convert



class hswish(nn.Module):
    def forward(self, x):
        out=x*torch.clamp(x+3,0,6) /6
#         out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)

        self.bn = nn.BatchNorm2d(out_channels)
        
        if self.if_act:
            if self.act == "hardswish":
                self.act_fun = hswish()
            else:
                self.act_fun = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act_fun(x)
        return x

class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        
        self.bottleneck_conv = RepVGGBlock(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride,  groups=mid_channels,deploy=False, use_se=use_se)
        if act=="relu":
            self.bottleneck_conv.nonlinearity = nn.ReLU()
        else:
            self.bottleneck_conv.nonlinearity = hswish()
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs+x
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = hsigmoid()

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs)
        return inputs * outputs

class MobileNetV3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        if small_stride is None:
            small_stride = [1, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', (large_stride[0],1)],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, False, 'relu', (large_stride[2], 1)],
                [5, 120, 40, False, 'relu', 1],
                [5, 120, 40, False, 'relu', 1],
                [5, 240, 80, False, 'hardswish', 1],
                [5, 200, 80, False, 'hardswish', 1],
                [5, 184, 80, False, 'hardswish', 1],
                [5, 184, 80, False, 'hardswish', 1],
                [5, 480, 112, False, 'hardswish', 1],
                [5, 672, 112, False, 'hardswish', 1],
                [5, 672, 160, False, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160, False, 'hardswish', 1],
                [5, 960, 160, False, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (small_stride[1], 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, False, 'hardswish', (small_stride[2], 1)],
                [5, 240, 40, False, 'hardswish', 1],
                [5, 240, 40, False, 'hardswish', 1],
                [5, 120, 48, False, 'hardswish', 1],
                [5, 144, 48, False, 'hardswish', 1],
                [5, 288, 96, False, 'hardswish', (small_stride[3], 1)],
                [5, 576, 96, False, 'hardswish', 1],
                [5, 576, 96, False, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        inplanes = 16
        # conv1
        self.conv1 = RepVGGBlock(in_channels=in_channels, out_channels=make_divisible(inplanes * scale), kernel_size=3, stride=2, groups=1,deploy=False, use_se=False)
        self.conv1.nonlinearity = hswish()
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x

    
def mobilenet_v3_small(pretrained,scale,is_gray=False):
    if is_gray:
        in_channels = 1
    else:
        in_channels = 3
    model = MobileNetV3( in_channels=in_channels,
                 model_name='small',
                 scale = scale)
    if pretrained:
        pass
    return model

def mobilenet_v3_large(pretrained,scale,is_gray=False):
    if is_gray:
        in_channels = 1
    else:
        in_channels = 3
    model = MobileNetV3( in_channels=in_channels,
                 model_name='large',
                 scale=scale)
    if pretrained:
        pass
    return model



# model1 = mobilenet_v3_small(False,scale=1)
# # model1 = mobilenet_v3_large(False,scale=1)
# print(model1)
# import torch
# img = torch.rand(1,3,32,320)
# out = model1(img)

# model2 = repvgg_model_convert(model1)
# print(model2)
# out = model2(img)
# print(out.shape)

# for key in model1.state_dict().keys():
#     print(key,model1.state_dict()[key].shape)

