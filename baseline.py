#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np  
from keras.models import Model,Sequential 
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input  
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.layers.merge import concatenate 
import matplotlib.pyplot as plt  
import cv2
import random
import os
import tensorflow as tf
from keras.utils import plot_model
from contextlib import redirect_stdout 
from tqdm import tqdm  
from keras.optimizers import Adam,SGD,RMSprop
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from math import pow,floor

from net.ICNet import build_bn

from keras import backend as K  
K.set_image_dim_ordering('tf')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # limit GPU_Memory
session = tf.Session(config=config)

seed = 7  
np.random.seed(seed)  
  
 
img_w = 256  
img_h = 256  
 
n_label = 6

classes = [0.,1.,2.,3.,4.,5.]  

  
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  

def load_img(path, grayscale = False):
    if grayscale:
        #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = Image.open(path)
    else:
        #img = cv2.imread(path)
        img = Image.open(path)
        img = np.array(img,dtype="float") / 255.0
    return img



filepath_train ='/storage/student17/rs/data/crop/Vaihingen/train/'
filepath_val ='/storage/student17/rs/data/crop/Vaihingen/val/'


#data for training and val
def get_train_val():  
    train_set = []
    val_set  = []
    for pic1 in os.listdir(filepath_train + 'src'):
        train_set.append(pic1)
    #random.shuffle(train_set)
    for pic2 in os.listdir(filepath_val + 'src'):
        val_set.append(pic2)
    #random.shuffle(val_set)
    return train_set,val_set
'''
def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set
'''

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []   
        train_label1, train_label2, train_label3 = [], [], []
        batch = 0  
        for i in (range(len(data))): 
            #url = data[i]
            batch += 1 
            img = load_img(filepath_train + 'src/' + data[i])
            img = img_to_array(img)
            #img = img[:,:,1:]
            train_data.append(img)  
            label = load_img(filepath_train + 'label/' + data[i], grayscale = True)
            #label = img_to_array(label)
            
            label1 = np.asarray(label.resize((img_h // 4, img_w // 4), Image.BILINEAR))
            label2 = np.asarray(label.resize((img_h // 8, img_w // 8), Image.BILINEAR))
            label3 = np.asarray(label.resize((img_h // 16, img_w // 16), Image.BILINEAR))
 
            train_label1.append(label1)
            train_label2.append(label2)
            train_label3.append(label3)
        
 
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)

                #train_label = np.array(train_label)          

                train_label1 = np.array(train_label1).flatten()    
                train_label1 = labelencoder.transform(train_label1)  
                train_label1 = to_categorical(train_label1, num_classes=n_label)  
                train_label1 = train_label1.reshape((batch_size,img_w // 4,img_h //4,n_label))

                train_label2 = np.array(train_label2).flatten()    
                train_label2 = labelencoder.transform(train_label2)  
                train_label2 = to_categorical(train_label2, num_classes=n_label)  
                train_label2 = train_label2.reshape((batch_size,img_w // 8,img_h // 8,n_label))
 
                train_label3 = np.array(train_label3).flatten()    
                train_label3 = labelencoder.transform(train_label3)  
                train_label3 = to_categorical(train_label3, num_classes=n_label)  
                train_label3 = train_label3.reshape((batch_size,img_w // 16,img_h // 16,n_label))

   
                yield (train_data,[train_label1,train_label2,train_label3])  
                train_data = []  
                train_label1,train_label2,train_label3 = [],[],[]
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label1, valid_label2, valid_label3 = [], [], []
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath_val + 'src/' + url)
            img = img_to_array(img)
            #img = img[:,:,1:] 
            valid_data.append(img)  
            label = load_img(filepath_val + 'label/' + url, grayscale = True)
            #label = img_to_array(label)

            label1 = np.asarray(label.resize((img_h // 4, img_w // 4), Image.BILINEAR))
            label2 = np.asarray(label.resize((img_h // 8, img_w // 8), Image.BILINEAR))
            label3 = np.asarray(label.resize((img_h // 16, img_w // 16), Image.BILINEAR))

           
            valid_label1.append(label1) 
            valid_label2.append(label2)
            valid_label3.append(label3)
  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)
  
                #valid_label = np.array(valid_label)

                valid_label1 = np.array(valid_label1).flatten()    
                valid_label1 = labelencoder.transform(valid_label1)  
                valid_label1 = to_categorical(valid_label1, num_classes=n_label)  
                valid_label1 = valid_label1.reshape((batch_size,img_w // 4,img_h // 4,n_label))

                valid_label2 = np.array(valid_label2).flatten()    
                valid_label2 = labelencoder.transform(valid_label2)  
                valid_label2 = to_categorical(valid_label2, num_classes=n_label)  
                valid_label2 = valid_label2.reshape((batch_size,img_w // 8,img_h // 8,n_label))

                valid_label3 = np.array(valid_label3).flatten()    
                valid_label3 = labelencoder.transform(valid_label3)  
                valid_label3 = to_categorical(valid_label3, num_classes=n_label)  
                valid_label3 = valid_label3.reshape((batch_size,img_w // 16,img_h // 16,n_label))


                yield (valid_data,[valid_label1,valid_label2,valid_label3])  
                valid_data = []  
                valid_label1,valid_label2,valid_label3 = [],[],[]
                batch = 0  

###########################################train###################################################

'''
def scheduler(epoch):
    init_lrate = 0.1
    drop = 0.5
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    print("lr changed to {}".format(lrate))
    return lrate
'''
 
def train(args):

    epochs = 150
    # bs = 4
    bs = 12
    
    #model = SegNet()  
    #model = unet(n_label = 6)
    #model = deeplabv3_plus(input_shape=(400,400,3),out_stride=16,num_classes=6)
    #model = ScasNet_VGG(n_label = 6)
    #model = ScasNet_ResNet(n_label = 6)
    #model = CE_net()
    #model = ce_net()
    model = build_bn(256, 256, 6, weights_path=None, train=True)
    
    model.summary()

    with open('./CheckPoint/baseline1/model_summary.txt', 'w') as f:    #########
           with redirect_stdout(f):
                  model.summary(line_length=200,positions=[0.30,0.60,0.7,1.0])

    ###################### learning_rate scheduler ###################### 
    
    def lr_scheduler(epoch):
         lr_base = 0.01
         lr_power = 0.9    
         lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
         #print('lr: %f' % lr)
         print("lr changed to {}".format(lr))
         return lr
    
    lrate = LearningRateScheduler(lr_scheduler)

    #callbacks_list = [lrate]
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

    ############################# optimizer #############################

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #rMSprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-4, decay=0.0)
    #sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss=['categorical_crossentropy' for _ in range(3)],loss_weights=[0.4, 0.4, 1,],optimizer = 'adam',metrics=['accuracy']) #1
     
    #ICNet loss_weights=[0.16, 0.4, 1]
    
    ############################# make model ##############################
    #modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')
    modelcheck = ModelCheckpoint('/storage/student17/rs/CheckPoint/baseline1/weights.{epoch:03d}-{val_conv6_cls_acc:.4f}-{val_conv6_cls_loss:.4f}.h5',monitor='val_conv6_cls_acc',save_best_only=True,mode='max') #####
    
  
    #callable = [modelcheck]  
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  

    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)

    H = model.fit_generator(generator=generateData(bs,train_set),steps_per_epoch=train_numb//bs,epochs=epochs,verbose=1,
                            validation_data=generateValidData(bs,val_set),validation_steps=valid_numb//bs,callbacks=[modelcheck,lrate],max_q_size=1)  

    plot_model(model, to_file='./CheckPoint/baseline1/model_polt.png',show_shapes = True)   ######

    ############################################################################
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["conv6_cls_loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_conv6_cls_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["conv6_cls_acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_conv6_cls_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

  

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--data", help="training data's path",
    #                default=True)
    #ap.add_argument("-m", "--model", required=True,
    #                help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="./CheckPoint/baseline1/plot.png",
                    help="path to output accuracy/loss plot")         ##########
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    #filepath = args['data']
    train(args)  
    #predict()  
