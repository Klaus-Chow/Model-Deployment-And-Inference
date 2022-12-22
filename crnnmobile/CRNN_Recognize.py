#coding: utf-8
import sys
sys.path.append('../')
import torch
from torch.autograd import Variable
import crnnmobile.utils as utils
import crnnmobile.dataset as dataset
import os
from PIL import Image
from keys_sythtext import alphabet
from models.repvgg import repvgg_model_convert
from crnnmobile.models.crnn_mobilev3_repvgg import CRNNSmallNoLSTM,CRNNLargeNoLSTM,CRNNSmallLSTM,CRNNLargeLSTM

class CRNN_Recognize():
    def __init__(self,model_path):
        super(CRNN_Recognize,self).__init__()
        nclass = len(alphabet) + 1
        

#         self.net = CRNNSmallLSTM(64,nclass,1).cuda()
        self.net = CRNNSmallNoLSTM(128,nclass,1).cuda()

        model_dict = torch.load(model_path)
        self.net = repvgg_model_convert(self.net,None)
        self.net.load_state_dict(model_dict)
        self.converter = utils.strLabelConverter(alphabet)
        self.transformer = dataset.resizeNormalize(32,280,'test')
        
    def recognize(self,image):
        image = self.transformer(image).cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        self.net.eval()
        with torch.no_grad():
            preds = self.net(image)
        preds_softmax = torch.softmax(preds,2).detach().cpu().numpy()
        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        confidence = []
        for i in range(preds.shape[0]):
            if(preds[i]!=0 and (not (i > 0 and preds[i - 1] == preds[i]))):
                confidence.append(float(preds_softmax[i,0,preds[i]]))
        preds = preds.transpose(0, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred,confidence
    
    def recognize_more(self,image):
        w,h = image.size
        tag = int(1/10*h)
        image = self.transformer(image,'yes',tag).cuda()
        image = Variable(image)

        self.net.eval()
        with torch.no_grad():
            preds_all = self.net(image)
        all_texts = []
        all_confidences = []
        for i in range(preds_all.shape[1]):
            preds = preds_all[:,i].unsqueeze(1)
            preds_softmax = torch.softmax(preds,2).detach().cpu().numpy()
            _, preds = preds.max(2)
            preds = preds.squeeze(1)
            confidence = []
            for i in range(preds.shape[0]):
                if(preds[i]!=0 and (not (i > 0 and preds[i - 1] == preds[i]))):
                    confidence.append(float(preds_softmax[i,0,preds[i]]))
            preds = preds.transpose(0, 0).contiguous().view(-1)

            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
            all_texts.append(sim_pred)
            all_confidences.append(confidence)
        return all_texts,all_confidences