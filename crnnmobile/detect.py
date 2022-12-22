#encoding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import cv2
import shutil
import numpy as np
import json
import time
import base64
from io import BytesIO
from tqdm import tqdm
from PIL import Image
sys.path.append('../')
from CRNN_Recognize import CRNN_Recognize

if __name__=="__main__":

    crnn_recognize = CRNN_Recognize("/src/notebooks/crnnmobile/model/mobile_fix.pth")
    text_path="/src/notebooks/MyWorkData/IDNEWDATA20220624Train/cece_train.txt"
    with open(text_path,'r') as fid:
        for line in fid.readlines():
            img_path=line.split(' ')[0]
            label=line.split(' ')[-1].strip()
            im_r = Image.open(img_path).convert('RGB') 
            import pdb
            pdb.set_trace()
            sim_pred,confidence=crnn_recognize.recognize(im_r)
            print("预测结果:"+sim_pred+" 标签："+label)
#             print(img_path)
#             print(label)

#             files.append(os.path.join(path,line.split('\t')[0].strip()))
    
    