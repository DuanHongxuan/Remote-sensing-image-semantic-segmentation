import cv2
import random
import numpy as np
import os,json
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder 
from PIL import Image
from tqdm import tqdm
import keras
import tensorflow as tf

import time


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


img_w = 256  
img_h = 256

classes = [0.,1.,2.,3.,4.,5.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument("-m", "--model", required=True,
    #    help="path to trained model model")
    #ap.add_argument("-s", "--stride", required=False,
    #    help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

def out_mask_to_color_pic(mask, palette_file='Palette.json'):
    assert len(mask.shape) == 2
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
    color_pic = np.zeros((mask.shape[0], mask.shape[1],3), dtype=np.uint8)
    for i in tqdm(range(0, mask.shape[0])):
        for j in range(0, mask.shape[1]):
            assert str(mask[i,j]) in list(text.keys())
            color_pic[i,j,:] = text[str(mask[i,j])]
    return color_pic


test_path = "/storage/student17/rs/data/source/Potsdam/test/src1"

def GetFileName(fileDir):
    list_name = []
    files = os.listdir(fileDir)
    for i in files:
        name = i.split('.')
        list_name.append(name[0])
    return list_name

files = GetFileName(test_path)
  
def predict(args):
    # load the trained convolutional neural network
    print("loading network...")
    #model = load_model("/storage/student17/RemoteSensing/Checkpoint/Mine/1/weights.037-0.8351--0.5134.h5")
    #model = load_model("/storage/student17/RemoteSensing/c/weights-051-0.8553-0.4907.h5", custom_objects={'BilinearUpsampling':BilinearUpsampling,'relu6':relu6})

    model = keras.models.load_model("/storage/student17/rs/CheckPoint/17/weights-075-0.8670-1.0175.h5", custom_objects={'tf': tf})

    stride = 256
    
    for n in tqdm(range(len(files))):
        #load the image
        #img = Image.open(test_path+'/'+files[n]+'.tif')
        img = cv2.imread(test_path+'/'+files[n]+'.tif')
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img = np.asarray(img)
        h,w,_ = img.shape
        #img = img.reshape(img.shape[0], img.shape[1],1, img.shape[2]) 
        img = img.astype("float") / 255.0
        #print(w,h,img.shape)
        mask_whole = np.zeros((h,w),dtype=np.uint8)
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
                img_crop = img[h_begin:h_end,w_begin:w_end]
                ch,cw,_ = img_crop.shape
                #print(cw,ch)
                if ch != img_h or cw != img_w:
                    print('invalid size!')
                    continue
             
                #img_crop = np.transpose(img_crop, (2, 0, 1))
                #print("1.",img_crop.shape)  
                img_crop = np.expand_dims(img_crop, axis=0)
                #print("2.",img_crop.shape)
                #start = time.clock()  
                pred = model.predict(img_crop,verbose=2)
                #end = time.clock() - start
                #time_sum = time_sum + end
                #print("3.",pred.shape)
                pred = pred[0]
                pred = np.reshape(pred, (1, 256 * 256, 6))
                pred = np.argmax(pred,axis=-1)
                #pred = pred.flatten()
                #print("3.",np.unique(pred)) 
                pred = labelencoder.inverse_transform(pred) 
                #print("4.",np.unique(pred))
                pred = pred.reshape((256,256)).astype(np.uint8)
                #pred = Bilinear(pred,4)
                #pred = cv2.resize(pred, (256, 256))
                mask_whole[h_begin:h_end,w_begin:w_end] = pred[:,:]
                
        out = out_mask_to_color_pic(mask_whole[0:h,0:w])
        #cv2.imwrite('./Predict/'+files[n]+'_label3.tif',mask_whole[0:h,0:w])
        Image.fromarray(out).save('./predict/17/'+files[n]+'.tif')
       
if __name__ == '__main__':
    start = time.clock()
    #time_sum = 0 
    args = args_parse()
    predict(args)
    print("Time:",time.clock() - start)
    #print("Time:",time_sum)
