import torch.nn as nn
import sys
sys.path.insert(0,'./')
from models.rec_mobilev3_repvgg import mobilenet_v3_small,mobilenet_v3_large,ConvBNLayer

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
        self.cnn = mobilenet_v3_small(False,scale,is_gray=False)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(int(576*scale), nh, nclass))
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

# class CRNNSmallLSTM(nn.Module):

#     def __init__(self,nh,nclass,scale):
#         super(CRNNSmallLSTM, self).__init__()
#         self.cnn = mobilenet_v3_small(False,scale,is_gray=False)
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(int(576*scale), nh, nh),
#             BidirectionalLSTM(nh, nh, nclass))
#     def forward(self, x):
#         # conv features
#         conv = self.cnn(x)

#         b, c, h, w = conv.size()
#         assert h == 1, "the height of conv must be 1"
#         conv = conv.squeeze(2)
#         conv = conv.permute(2, 0, 1)  # [w, b, c]

#         # rnn features
#         output = self.rnn(conv)

#         return output
    
class CRNNLargeLSTM(nn.Module):

    def __init__(self,nh,nclass,scale):
        super(CRNNLargeLSTM, self).__init__()
        self.cnn = mobilenet_v3_large(False,scale,is_gray=False)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(int(960*scale), nh, nh),
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
        self.cnn = mobilenet_v3_small(False,scale,is_gray=False)
        self.convDown = ConvBNLayer(
            in_channels=int(576*scale),
            out_channels=nh,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act='hardswish')
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
#         assert h == 1, "the height of conv must be 1"
#         conv = conv.squeeze(2)
        conv = conv.view(-1,c,h*w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        return conv
    
class CRNNLargeNoLSTM(nn.Module):

    def __init__(self,nh,nclass,scale):
        super(CRNNLargeNoLSTM, self).__init__()
        self.cnn = mobilenet_v3_large(False,scale,is_gray=False)
        self.convDown = ConvBNLayer(
            in_channels=int(960*scale),
            out_channels=nh,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act='hardswish')
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
# model = CRNNSmallNoLSTM(128,5000,1)
# model = CRNNLargeNoLSTM(128,5000,1)
# model = CRNNSmallLSTM(128,5000,1)
# model = CRNNLargeLSTM(128,5000,1)
# torch.save(model.state_dict(),'1.pt')
# print(model)
# out = model(img)
# print(out.shape)