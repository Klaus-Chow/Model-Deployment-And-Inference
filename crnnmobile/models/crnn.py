import torch.nn as nn
import sys
sys.path.append('./crnn/models/')
#from resnet import *
# from resnet_v import *
# from resnet_v_v import *
from resnet_v_o import *
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

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = resnet50(pretrained=False)
        self.rnn = nn.Sequential(
            #BidirectionalLSTM(2048, nh, nh),
#             BidirectionalLSTM(2944+64, nh, nh),
            BidirectionalLSTM(2944, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # conv out
        #conv = self.cnn_out(conv)
        
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        #output = conv
        # rnn features
        output = self.rnn(conv)

        return output
