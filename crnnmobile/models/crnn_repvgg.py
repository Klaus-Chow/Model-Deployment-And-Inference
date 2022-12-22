import torch.nn as nn
from crnn.models.repvgg import get_RepVGG_func_by_name
# from crnn.models.rec_mobilev3 import ConvBNLayer
import torch.nn.functional as F

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
        if self.act == "hardswish":
            self.hardswish = hswish()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = self.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x
    

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

## 576,960
class CRNNSmallLSTM(nn.Module):

    def __init__(self,nh,nclass,scale):
        super(CRNNSmallLSTM, self).__init__()
        repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A0')
        self.cnn = repvgg_build_func(deploy=False)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1280, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
    def forward(self, x):
        # conv features
        conv = self.cnn(x)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
    
class CRNNLargeLSTM(nn.Module):

    def __init__(self,nh,nclass,scale):
        super(CRNNLargeLSTM, self).__init__()
        repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A1')
        self.cnn = repvgg_build_func(deploy=False)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1280, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
    def forward(self, x):
        # conv features
        conv = self.cnn(x)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output
    
class CRNNSmallNoLSTM(nn.Module):

    def __init__(self,nh,nclass,scale):
        super(CRNNSmallNoLSTM, self).__init__()
        repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A0')
        self.cnn = repvgg_build_func(deploy=False)
        self.convDown = ConvBNLayer(
            in_channels=1280,
            out_channels=nh,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act='relu')
        self.convClass = nn.Conv2d(
            in_channels=nh,
            out_channels=nclass,
            kernel_size=1,
            stride=1,
            padding=0)
    def forward(self, x):
        # conv features
        conv = self.cnn(x)
        conv = self.convClass(self.convDown(conv))
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        return conv
    
class CRNNLargeNoLSTM(nn.Module):

    def __init__(self,nh,nclass,scale):
        super(CRNNLargeNoLSTM, self).__init__()
        repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A1')
        self.cnn = repvgg_build_func(deploy=False)
        self.convDown = ConvBNLayer(
            in_channels=1280,
            out_channels=nh,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act='relu')
        self.convClass = nn.Conv2d(
            in_channels=nh,
            out_channels=nclass,
            kernel_size=1,
            stride=1,
            padding=0)
    def forward(self, x):
        # conv features
        conv = self.cnn(x)
        conv = self.convClass(self.convDown(conv))

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        return conv

# import torch
# img = torch.rand(1,3,32,320)
# # model = CRNNSmallNoLSTM(128,5000,1)
# # model = CRNNLargeNoLSTM(128,5000,1)
# # model = CRNNSmallLSTM(128,5000,1)
# model = CRNNLargeLSTM(128,5000,1)
# torch.save(model.state_dict(),'1.pt')
# print(model)
# out = model(img)
# print(out.shape)