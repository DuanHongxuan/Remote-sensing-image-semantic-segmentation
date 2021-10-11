from PIL import Image,ImageFilter,ImageDraw,ImageEnhance
import random
import os
import numpy as np
from tqdm import tqdm
import cv2


img_w = 256  
img_h = 256  

#train_path = "/storage/student17/RemoteSensing/data/source/Vaihingen/train/src"
#labels_path = "/storage/student17/RemoteSensing/data/source/Vaihingen/train/new_label"

train_path = "/storage/student17/rs/data/source/Potsdam/train/src1"
labels_path = "/storage/student17/rs/data/source/Potsdam/train/new_label"


def GetFileName(fileDir):
    list_name = []
    files = os.listdir(fileDir)
    for i in files:
        name = i.split('.')
        list_name.append(name[0])
    return list_name

files = GetFileName(train_path)

########################data_augment################################
'''
def add_noise(img):
    drawObject=ImageDraw.Draw(img)
    for i in range(250): 
        temp_x = np.random.randint(0,img.size[0])
        temp_y = np.random.randint(0,img.size[1])
        drawObject.point((temp_x,temp_y),fill="white")
    return img

def random_color(img):
    img = ImageEnhance.Color(img)
    img = img.enhance(2)
    return img

def data_augment(src_roi,label_roi):
    if np.random.random() < 0.25:
       src_roi=src_roi.rotate(90)
       label_roi=label_roi.rotate(90)
    if np.random.random() < 0.25:
       src_roi=src_roi.rotate(180)
       label_roi=label_roi.rotate(180)
    if np.random.random() < 0.25:
       src_roi=src_roi.rotate(270)
       label_roi=label_roi.rotate(270)
     
    if np.random.random() < 0.25:
       src_roi=src_roi.transpose(Image.FLIP_LEFT_RIGHT)
       label_roi=label_roi.transpose(Image.FLIP_LEFT_RIGHT)
   
    if np.random.random() < 0.25:
       src_roi=src_roi.transpose(Image.FLIP_TOP_BOTTOM)
       label_roi=label_roi.transpose(Image.FLIP_TOP_BOTTOM)
   
    if np.random.random() < 0.25:
       src_roi=src_roi.filter(ImageFilter.GaussianBlur)
    
    if np.random.random() < 0.25:
       src_roi=random_color(src_roi)
      
    if np.random.random() < 0.2:
       src_roi = add_noise(src_roi)
    return src_roi,label_roi
'''
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): 
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1) 
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb
################################creat_dataset#######################################   

def creat_dataset(mode = 'original'):
    print('creating dataset...')
    g_count = 0
    stride = 64     
    for i in tqdm(range(len(files))):
        count = 0
        #src_img = Image.open(train_path+'/'+files[i]+'.tif')  # 3 channels
        src_img = cv2.imread(train_path+'/'+files[i]+'.tif')                         
        #label_img = Image.open(labels_path+'/'+files[i]+'_new.tif')
        label_img = cv2.imread(labels_path+'/'+files[i]+'_new.tif')
         
        print("name:",files[i])
        print("size:",src_img.shape)
        
        w,h,_ = src_img.shape
        while count < 1:
             for i in range(h // stride + 1):
                 for j in range(w // stride + 1):
                     #print(i,j)
                     h_begin = i*stride
                     w_begin = j*stride
                     #print('h_begin:',h_begin,'w_begin:',w_begin)
                     if h_begin + img_h > h:
                         h_begin = h_begin - (h_begin + img_h - h)
                     if w_begin + img_w > w:
                         w_begin = w_begin - (w_begin + img_w - w)
                     w_end = w_begin + img_w
                     h_end = h_begin + img_h
                     #print("w_end:",w_end,"h_end:",h_end)
                     #print(w_begin, h_begin, w_end, h_end)
                     #src_crop = src_img.crop((w_begin, h_begin, w_end, h_end))
                     #label_crop = label_img.crop((w_begin, h_begin, w_end, h_end))
                     src_crop = src_img[w_begin:w_end, h_begin:h_end,:]
                     label_crop = label_img[w_begin:w_end, h_begin:h_end]  
                     if mode == 'augment':
                         src_crop,label_crop = data_augment(src_crop,label_crop)

                     #src_crop.save('/storage/student17/rs/data/crop/Potsdam/train/src/%d.tif' % g_count)
                     #label_crop.save('/storage/student17/rs/data/crop/Potsdam/train/label/%d.tif' % g_count)
                     cv2.imwrite(('/storage/student17/rs/data/crop/Potsdam/train/src/%d.tif' % g_count),src_crop)
                     cv2.imwrite(('/storage/student17/rs/data/crop/Potsdam/train/label/%d.tif' % g_count),label_crop)
                     g_count += 1
             print("number:",g_count)
             
             print("count:",count)
             count +=1
        print("end")
             
                     

if __name__=='__main__':  
    creat_dataset(mode='augment')

