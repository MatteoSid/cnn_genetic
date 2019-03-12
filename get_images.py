from os import listdir
from PIL import Image, ImageChops
import random
from os.path import isfile, join
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
from tqdm import tqdm

# libreria per salvare l'array in forato .npy
from tempfile import TemporaryFile
outfile = TemporaryFile()

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

        print('Verranno usati i seguenti dataset:\n' + selection_path + '\n' + neutral_path)

    elif mode == 'TEST':
        print('Script avviato in modalità TEST\n')
        selection_path = files_path + 'SELECTION/TEST_IMG/'
        neutral_path = files_path + 'NEUTRAL/TEST_IMG/'

        print('Verranno usati i seguenti dataset:\n' + selection_path + '\n' + neutral_path)

    images_arr = []     # lista che conterrà tutte le immagini
    label_arr = []      # lista che conterrà tutte le etichette

    # creo un file per il log
    log = open(files_path + '/log.txt', 'w')
    log = open(files_path + '/log.txt', 'a')
    log.write('Log di caricamento:')

    # Carico il SELECTION dataset [0,1]
    files = [f for f in listdir(selection_path) if isfile(join(selection_path, f))]
    log.write('\n\nCARICAMENTO SELECTION')
    print('\nCarico il SELECTION dataset:')
    for fl in tqdm(files):
        if fl != '.DS_Store':

            #print('\nloading file: ' + fl)  # stampo il file corrente
            label_arr.append([0,1])         # applico l'etichetta [0,1] a tutte le immagini SELECTION

            # carico le immagini convertendole in immagini ad un canale
            image = Image.open(selection_path + fl).convert(mode='1')    

            log.write('\n\nImage raw:\n')
            log.write(str(image))
            
            image = image.resize((img_size_w,img_size_h))
            #image.show()
            
            log.write('\n\nImage resized:\n')
            log.write(str(image))

            images_arr.append(np.array(image,dtype=int))
            
    # Carico il NEUTRAL dataset [1,0]
    files = [f for f in listdir(neutral_path) if isfile(join(neutral_path, f))] 
    log.write('\n\n\nCARICAMENTO NEUTRAL')
    print('\nCarico il neutral dataset:')
    for fl in tqdm(files):
        if fl != '.DS_Store':

            #print('\nloading file: ' + fl)  # stampo il file corrente
            label_arr.append([1,0])         # applico l'etichetta [0,1] a tutte le immagini NEUTRAL

            # carico le immagini convertendole in immagini ad un canale
            image = Image.open(neutral_path + fl).convert(mode='1')

            log.write('\n\nImage raw:\n')
            log.write(str(image))

            image = image.resize((img_size_w, img_size_h))
            #image.show()

            log.write('\n\nImage resized:\n')
            log.write(str(image))

            images_arr.append(np.array(image, dtype=int))
        
    # trasformo le liste in array numpy
    images_arr = np.array(images_arr)
    label_arr = np.array(label_arr)

    print('\nGli array hanno dimensione:\n  -images_arr: ' + str(len(images_arr)) + '\n  -label_arr: ' + str(len(label_arr)))
    print('\nimages_array shape: ' + str(images_arr.shape))

    print('label_array shape: ' + str(label_arr.shape))
    
    ####### LOG #######
    log.write('\n\nGli array hanno dimensione:\n  -images_arr: ' + str(len(images_arr)) + '\n  -label_arr: ' + str(len(label_arr)))
    log.write('\n\nimages_array shape: ' + str(images_arr.shape))
    log.write('\nlabel_array shape: ' + str(label_arr.shape))
    #log.write('\n\nImages:\n\n' + str(images_arr))
    ##################

    images_arr = images_arr.reshape(len(images_arr),n_input)
    print('images_arr reshaped: ' + str(images_arr.shape) + '\n')
    log.write('\nimages_arr reshaped: ' + str(images_arr.shape) + '\n')
    log.write('\n\n\nLabels:\n\n' + str(label_arr))
    
    log.write('\n\n' + str(np.array2string(images_arr, threshold=np.nan, max_line_width=np.nan)))

    #log.write('\n\nRESHAPED:\n' + str(images_arr))
    log.close()

    # np.save(files_path, images_arr) # serve per salvare un array su disco (vedi anche savetxt)
    return len(images_arr),images_arr,label_arr


""" MAIN """
os.system('clear')
files_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'
img_size_w = 48
img_size_h = 1000
n_input = 48000 # w*h
mode = 'TRAIN'

get_images(
    files_path=files_path,
    img_size_w=img_size_w,
    img_size_h=img_size_h,
    n_input=n_input,
    mode=mode)

"""
Image.resize(size, resample=0)
    Returns a resized copy of this image.

    Parameters:
    size – The requested size in pixels, as a 2-tuple: (width, height).
    resample – An optional resampling filter. This can be one of PIL.Image.NEAREST(use nearest neighbour), PIL.Image.BILINEAR(linear interpolation), PIL.Image.BICUBIC(cubic spline interpolation), or PIL.Image.LANCZOS(a high-quality downsampling filter). If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
    Returns:
    An Image object.
"""
