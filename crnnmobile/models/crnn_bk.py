import torch.nn as nn
import torch.nn.functional as F
import torch
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

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        stage1 = nn.Sequential()
        stage2 = nn.Sequential()
        stage3 = nn.Sequential()
        stage4 = nn.Sequential()
        stage5 = nn.Sequential()

        def convRelu(cnn,i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(stage1,0)
        stage1.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(stage2,1)
        stage2.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(stage3,2, True)
        convRelu(stage3,3)
        stage3.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(stage4,4, True)
        convRelu(stage4,5)
        stage4.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(stage5,6, True)  # 512x1x16

        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4
        self.stage5 = stage5
        self.wconv1 =  nn.Conv2d(128, 128,(8,1))
        self.wconv1_bn = nn.BatchNorm2d(128)
        self.wconv1_relu = nn.ReLU()
        self.wconv2 =  nn.Conv2d(256, 128,(4,1))
        self.wconv2_bn = nn.BatchNorm2d(128)
        self.wconv2_relu = nn.ReLU()
        self.wconv3 =  nn.Conv2d(512, 256,(2,1))
        self.wconv3_bn = nn.BatchNorm2d(256)
        self.wconv3_relu = nn.ReLU()
#         self.cnn_out = nn.Conv2d(512,nclass,1)
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(512, nh, nclass))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
    def forward(self, input):
        # conv features
        conv = self.stage1(input)
#         print(conv.shape)
        conv1 = self.stage2(conv)
#         print('conv1',conv1.shape)

        _,_,h1,w1 = conv1.shape
        
         ####
        out1 = self.wconv1_relu(self.wconv1_bn(self.wconv1(conv1)))
        ####
        
        conv2 = self.stage3(conv1)
        _,_,h2,_ = conv2.shape
        conv2 = F.interpolate(conv2,(h2,w1))
#         print('conv2',conv2.shape)
        
        ####
        out2 = self.wconv2_relu(self.wconv2_bn(self.wconv2(conv2)))
        ####
        
        conv3 = self.stage4(conv2)
        _,_,h3,_ = conv3.shape
        conv3 = F.interpolate(conv3,(h3,w1))
#         print('conv3',conv3.shape)
        
        ####
        out3 = self.wconv3_relu(self.wconv3_bn(self.wconv3(conv3)))
        ####
        
        conv = self.stage5(conv3)
        _,_,h4,_ = conv.shape
        conv = F.interpolate(conv,(h4,w1))
     
        
        # conv out
        #conv = self.cnn_out(conv)
        
        conv = torch.cat((out1,out2,out3,conv),1)
#         conv = out1+out2+out3+conv

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        #output = conv
        # rnn features
        output = self.rnn(conv)

        return output

