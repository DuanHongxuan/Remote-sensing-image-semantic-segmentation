import json,os
import numpy as np
import tifffile as tiff
from PIL import Image 

os.environ["OMP_NUM_THREADS"] = "1"

def get_label_from_palette(label_img, palette_file='Palette.json'):
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
        label = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
        for i in range(label_img.shape[0]):
            #print(i)
            for j in range(label_img.shape[1]):
                
                assert list(label_img[i, j, :]) in list(text.values())

                label[i, j] = int(list(text.keys())[list(text.values()).index(list(label_img[i, j, :]))])

        return label

def main(path):
    for pic in os.listdir(path):
        
            print(pic)
            # ---- read RGB label
            label = Image.open(path + '/' +pic)
            label = np.asarray(label)
            # ----- another way 
            # label = tiff.imread(path + '/' +pic) 
            
            label = get_label_from_palette(label)
            tiff.imsave('/storage/student17/rs/data/source/Potsdam/test/new_label/'  +pic[:-4] + '_new.tif',label)
            


if __name__ == '__main__':
    #train_path = '/storage/student17/RemoteSensing/data/source/Vaihingen/train/label'
    #val_path = '/storage/student17/RemoteSensing/data/source/Vaihingen/val/label'
    test_path ='/storage/student17/rs/data/source/Potsdam/test/label'
    #main(train_path)
    #main(val_path)
    main(test_path)



