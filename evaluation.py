import os
import cv2
import numpy as np
import skimage.io as io
from PIL import Image
from tqdm import tqdm


path1 = "/storage/student17/rs/data/source/Vaihingen/evalution" #Dir of Ground Truth

path2 = "/storage/student17/rs/predict/7" #Dir of predict map


sample1 = os.listdir(path1)
'''
def GetFileName(fileDir):
    list_name = []
    files = os.listdir(fileDir)
    for i in files:
        name = i.split('.')
        list_name.append(name[0])
    return list_name

files = GetFileName(path1)
print(files)
for i in tqdm(range(len(files))):
   segmentationImage = io.imread(path2+'/'+files[i]+'.tif')
   gtLabelsImage = io.imread(path1+'/'+files[i]+'.tif')
'''

Iou = []#Iou for each test images
TP = 0
FP = 0
FN = 0
sum_fenmu = 0
sum_F1 = 0
sum_IoU = 0
sum_Pre = 0
for name in sample1:
    print("name:",name)
    mask1 = io.imread(os.path.join(path1, name))
    mask1 = mask1 / 255
    mask1 = mask1.flatten()
    
    #name1 = name[0:-8]+'sat.jpg'
    #mask2 = io.imread(os.path.join(path2, name1))
    mask2 = io.imread(os.path.join(path2, name))
    mask2 = mask2 / 255.0
    #mask2[mask2 >= 0.5] = 1
    #mask2[mask2 < 0.5] = 0
    mask2 = mask2.flatten()
    
    tp = np.dot(mask1, mask2)
    TP = TP + tp
    fp = mask2.sum()-tp
    FP = FP + fp
    fn = mask1.sum()-tp
    FN = FN + fn
    #fenmu = mask1.sum()+mask2.sum()-tp
    fenmu = mask1.sum()+mask2.sum()-tp
    sum_fenmu = sum_fenmu + fenmu
    #element_wise = np.multiply(mask1, mask2)
    Iou.append(tp / fenmu)
    #if(tp / fenmu == 0.0):
        #print(name)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    
    F1 = 2*precision*recall/(precision+recall)
    
    
    #print("Iou:",Iou)
    print("IoU:",TP / sum_fenmu)#active IoU
    #print("recall",TP / (TP+FN))#recall
    print("precision:",TP / (TP+FP))#precision
    print("F1:",F1) #F1 Score
    
    sum_IoU = TP/sum_fenmu + sum_IoU
    sum_Pre = TP/(TP+FP) + sum_Pre
    sum_F1 = F1 + sum_F1
    

print("==============================")
print("IoU:",sum_IoU / 17)#active IoU
#print("recall",TP / (TP+FN))#recall
print("precision:",sum_Pre / 17)#precision
print("F1:",sum_F1 / 17) #F1 Score



