from os import listdir
from PIL import Image, ImageChops
import random
from os.path import isfile, join
import numpy as np
import os

# FUNZIONE PER REPERIRE SET DI IMMAGINI DAL FILE SYSTEM
# Funziona sia per il training set che per ogni altro set

# La funzione prende in input:
# - files_path: il path del DATASET contenente tutte le immagini
# - img_size_w e img_size_h: rispettivamente larghezza e altezza dell'immagine
# - n_input: wxhxc dove w ed h sono larghezza ed altezza dell'immagine e c è il numero di canali dellimmagine (nel nostro caso c=1)

def get_images(files_path, img_size_w, img_size_h, n_input, mode):
    if mode == 'TRAIN':
        print('Script avviato in modalità TRAIN\n')
        selection_path = files_path + 'SELECTION/TRAIN_IMG/'
        neutral_path = files_path + 'NEUTRAL/TRAIN_IMG/'

        print('Verranno usati i seguenti dataset:')
        print(selection_path)
        print(neutral_path)

    images_arr = []
    label_arr = []

    # Carico il SELECTION dataset [0,1]
    files = [f for f in listdir(selection_path) if isfile(join(selection_path, f))]
    for fl in files:
        if fl != '.DS_Store':
            #print(fl)
            label_arr.append([0,1])
            image = Image.open(selection_path + fl)
            image = image.resize((img_size_w,img_size_h))
            images_arr.append(np.array(image,dtype=float)) 

    
     # Carico il NEUTRAL dataset [1,0]
    files = [f for f in listdir(neutral_path) if isfile(join(neutral_path, f))] 

    for fl in files:
        if fl != '.DS_Store':
            #print(fl)
            label_arr.append([1,0])
            image = Image.open(neutral_path + fl)
            image = image.resize((img_size_w, img_size_h))
            images_arr.append(np.array(image, dtype=float))
        

    images_arr = np.array(images_arr)
    label_arr = np.array(label_arr)

    print('Gli array hanno dimensione: ' + str(len(images_arr)) + '\t' + str(len(label_arr)))
    images_arr = images_arr.reshape(len(images_arr),n_input)
    # return len(images_arr),images_arr,label_arr


# Image.resize(size, resample=0)
#     Returns a resized copy of this image.

#     Parameters:
#     size – The requested size in pixels, as a 2-tuple: (width, height).
#     resample – An optional resampling filter. This can be one of PIL.Image.NEAREST(use nearest neighbour), PIL.Image.BILINEAR(linear interpolation), PIL.Image.BICUBIC(cubic spline interpolation), or PIL.Image.LANCZOS(a high-quality downsampling filter). If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
#     Returns:
#     An Image object.

os.system('clear')
files_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'
img_size_w = 48
img_size_h = 1000
n_input = 48000
# n_input = 3840000
mode = 'TRAIN'

get_images(
    files_path=files_path,
    img_size_w=img_size_w,
    img_size_h=img_size_h,
    n_input=n_input,
    mode=mode)
