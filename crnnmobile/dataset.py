#-*- coding:utf-8 _*-
"""
@author:fxw
@file: dataset.py
@time: 2019/07/10
"""
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import sys
from PIL import Image
import numpy as np
import os
import cv2
def Add_Padding(image, top, bottom, left, right, color=(0,0,0)):
    if(not isinstance(image,np.ndarray)):
        image = np.array(image)
    padded_image = cv2.copyMakeBorder(image, top, bottom,left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

class LoadDataset(Dataset):
    def __init__(self,image_path, file_txt=None, transform=None, target_transform=None):
        with open(file_txt,'r',encoding='utf-8') as fid:
            self.label_list = []
            self.image_list = []
            for line in fid.readlines():
                label = line.split('\t')[-1].strip().replace('\ufeff','')
                image_file = line.split('\t')[0].strip('\n').strip('\r\n').replace('\ufeff','')
                self.label_list.append(label)
                self.image_list.append(os.path.join(image_path,image_file))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):

        img = Image.open(self.image_list[index]).convert('L')
        label = self.label_list[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

def get_batch_images(image,tag):
    w, h = image.size
    ### cut top
    image_cut_top = image.crop((0, tag, w, h))
    new_w = int(32/(h-tag)*w)
    if(new_w<=0):
        new_w = w
    image_cut_top = image_cut_top.resize((new_w,32))
    ### cut bottom
    image_cut_bottom = image.crop((0, 0, w, h-tag))
    new_w = int(32/(h-tag)*w)
    if(new_w<=0):
        new_w = w
    image_cut_bottom = image_cut_bottom.resize((new_w,32))
    ###
    image = np.array(image)
    image = Add_Padding(image, 0, 0, 0, new_w-w)
    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    image_cut_top = transforms.ToTensor()(image_cut_top)
    image_cut_bottom = transforms.ToTensor()(image_cut_bottom)
    image.sub_(0.5).div_(0.5)
    image_cut_top.sub_(0.5).div_(0.5)
    image_cut_bottom.sub_(0.5).div_(0.5)
    img = torch.cat((image.unsqueeze(0),image_cut_top.unsqueeze(0),image_cut_bottom.unsqueeze(0)),0)
#     img = torch.cat((image.unsqueeze(0),image_cut_bottom.unsqueeze(0)),0)
    return img

class resizeNormalize(object):

    def __init__(self,height=32,max_width=280,types='train'):

        self.toTensor = transforms.ToTensor()
        self.max_width = max_width
        self.types = types
        self.height = height
    
    def __call__(self, img,itype = 'no',itag=0):
        if(self.types=='train' or self.types=='val'):
            w,h = img.size
            new_w = int(self.height/h*w)
            img = img.resize((new_w,self.height), Image.BILINEAR)
            if(new_w < self.max_width):
                img = Add_Padding(img, 0, 0, 0, self.max_width-new_w)
                img = Image.fromarray(img)
            else:
                img = img.resize((self.max_width,self.height), Image.BILINEAR)
            
        else:
            w,h = img.size
            img = img.resize((int(self.height/float(h)*w),self.height), Image.BILINEAR)
            if(itype=='yes'):
                img = get_batch_images(img,itag)
                return img
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

#     def __call__(self, img):
#         if(self.types=='train' or self.types=='val'):
#             w,h = img.size
#             if(w < self.max_width):
#                 img = Add_Padding(img, 0, 0, 0, self.max_width-w)
#                 img = Image.fromarray(img)
#             else:
#                 img = img.resize((self.max_width,self.height), Image.BILINEAR)
#         else:
#             w,h = img.size
#             img = img.resize((int(self.height/float(h)*w),self.height), Image.BILINEAR)
#         img = self.toTensor(img)
#         img.sub_(0.5).div_(0.5)
#         return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=256, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            new_images = []
            for image in images:
                w, h = image.size
                image = image.resize((int(imgH/float(h)*w),imgH), Image.BILINEAR)
                new_images.append(image)
            
        transform = resizeNormalize(32,280,'train')
        images = [transform(image) for image in new_images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
