#!usr/bin/env python
#-*- coding:utf-8 _*-
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance
import random
import os
import numpy as np
from tqdm import tqdm
import cv2

img_w = 256 
img_h = 256  

train_path = "/storage/student17/rs/data/source/Potsdam/val/src1"
labels_path = "/storage/student17/rs/data/source/Potsdam/val/new_label"

def GetFileName(fileDir):
    list_name = []
    files = os.listdir(fileDir)
    for i in files:
        name = i.split('.')
        list_name.append(name[0])
    return list_name

files = GetFileName(train_path)

def creat_dataset():
    print('creating dataset...')
    #image_each = image_num / len(files)
    g_count = 0
    stride = 64
    for i in tqdm(range(len(files))):
        count = 0
        #src_img = Image.open(train_path+'/'+files[i]+'.tif')  # 3 channels
        #src_img = img1.convert('CMYK')
        #label_img = Image.open(labels_path+'/'+files[i]+'_new.tif') # 3 channels
        #label_img = img2.convert('L')
        src_img = cv2.imread(train_path+'/'+files[i]+'.tif')
        label_img =cv2.imread(labels_path+'/'+files[i]+'_new.tif')
        print("name:",files[i])
        print("size:",src_img.shape)
        w,h,_ = src_img.shape
 
        for i in range(h // stride + 1):
            for j in range(w // stride + 1):

                h_begin = i*stride
                w_begin = j*stride
                if h_begin + img_h > h:
                    h_begin = h_begin - (h_begin + img_h - h)
                if w_begin + img_w > w:
                    w_begin = w_begin - (w_begin + img_w - w)
                w_end = w_begin + img_w
                h_end = h_begin + img_h
                #src_crop = src_img.crop((w_begin, h_begin, w_end, h_end))
                #label_crop = label_img.crop((w_begin, h_begin, w_end, h_end))
                src_crop = src_img[w_begin:w_end, h_begin:h_end,:]
                label_crop = label_img[w_begin:w_end, h_begin:h_end]
            
                #src_crop.save('/storage/student17/rs/data/crop/Vaihingen/val/src/%d.tif' % g_count)
                #label_crop.save('/storage/student17/rs/data/crop/Vaihingen/val/label/%d.tif' % g_count)
                cv2.imwrite(('/storage/student17/rs/data/crop/Potsdam/val/src/%d.tif' % g_count),src_crop)
                cv2.imwrite(('/storage/student17/rs/data/crop/Potsdam/val/label/%d.tif' % g_count),label_crop)
                count += 1 
                g_count += 1
        print("number:",g_count)
             
        print("count:",count)

if __name__=='__main__':  
    creat_dataset()

