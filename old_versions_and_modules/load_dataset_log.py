from os import listdir
from PIL import Image, ImageChops
import random
from os.path import isfile, join
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
from tqdm import tqdm

# setto la larghezza massima di stampa degli array numpy
np.set_printoptions(linewidth=10000)

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
    log = open(files_path + 'log.txt', 'w')
    log = open(files_path + 'log.txt', 'a')
    log.write('Log di caricamento:')

    # Carico il SELECTION dataset [0,1]
    files = [f for f in listdir(selection_path) if isfile(join(selection_path, f))]
    print('\nCarico il SELECTION dataset:')
    i = 0
    for fl in tqdm(files):
        if fl != '.DS_Store':
            #print('\nloading file: ' + fl)  # stampo il file corrente
            
            # carico le immagini convertendole in immagini ad un canale
            image = Image.open(selection_path + fl).convert(mode='L')  

            # Controllo che le immagini abbiano tutte la stessa dimensione, altrimenti le salto
            if str(image.size) == '(' + str(img_size_w) + ', ' + str(img_size_h) + ')':
                label_arr.append([0, 1, i])    # applico l'etichetta [0,1] a tutte le immagini SELECTION
                #print('size: ' + str(image.size))
                # log.write('\n\nImage raw:\n')
                # log.write(str(image))
                # log.write('\n\nImage resized:\n')
                # log.write(str(image))

                image_ID = np.full((img_size_h, 1), i)
                image = np.concatenate((image, image_ID), axis=1)

                images_arr.append(np.array(image,dtype=int))
                i = i +1
            
    # Carico il NEUTRAL dataset [1,0]
    files = [f for f in listdir(neutral_path) if isfile(join(neutral_path, f))] 
    print('\nCarico il NEUTRAL dataset:')
    for fl in tqdm(files):
        if fl != '.DS_Store':
            #print('\nloading file: ' + fl)  # stampo il file corrente

            # carico le immagini convertendole in immagini ad un canale
            image = Image.open(neutral_path + fl).convert(mode='L')
            
            # Controllo che le immagini abbiano tutte la stessa dimensione, altrimenti le salto
            if str(image.size) == '(' + str(img_size_w) + ', ' + str(img_size_h) + ')':
                label_arr.append([1,0,i])         # applico l'etichetta [0,1] a tutte le immagini NEUTRAL
                # log.write('\n\nImage raw:\n')
                # log.write(str(image))
                # log.write('\n\nImage resized:\n')
                # log.write(str(image))

                image_ID = np.full((img_size_h,1), i)
                image = np.concatenate((image, image_ID), axis=1)

                images_arr.append(np.array(image, dtype=int))
                i = i +1
        
    # trasformo le liste in array numpy
    images_arr = np.array(images_arr)
    label_arr = np.array(label_arr)

    print('\nGli array hanno dimensione:\n  -images_arr: ' + str(len(images_arr)) + '\n  -label_arr: ' + str(len(label_arr)))
    print('\nimages_array shape: ' + str(images_arr.shape))

    print('label_array shape: ' + str(label_arr.shape))
    
    ####### LOG #######
    log.write('\n\nGli array hanno dimensione:\n  - images_arr: ' + str(len(images_arr)) + '\n  - label_arr: ' + str(len(label_arr)))
    log.write('\n\nimages_array shape: ' + str(images_arr.shape))
    log.write('\nlabel_array shape: ' + str(label_arr.shape))
    #log.write('\n\nImages:\n\n' + str(images_arr))
    ##################

    #images_arr = images_arr.reshape(len(images_arr),n_input)

    ####### LOG #######
    #print('images_arr reshaped: ' + str(images_arr.shape) + '\n')
    #log.write('\nimages_arr reshaped: ' + str(images_arr.shape) + '\n')
    #log.write('\n\n\nLabels:\n\n' + str(label_arr))
    #log.write('\n\nImages:\n\n' + str(np.array2string(images_arr, threshold=np.nan, max_line_width=np.nan)))
    #log.write('\n\nRESHAPED:\n' + str(images_arr))
    log.close()
    ##################

    # np.save(files_path, images_arr) # serve per salvare un array su disco (vedi anche savetxt)
    return len(images_arr),images_arr,label_arr

# FUNZIONE PER IL RECUPERO DI UN BATCH SI IMMAGINI DA UN SET

# La funzione prende in input:
# - total: il numero totale di elementi del dataset
# - images: array contenente tutte le immagini
# - labels: array contentenente tutte le etichette
# - batch_size: dimensione dei singoli batch da creare
# - index: indice del batch da estrarre
def next_batch(total, images, labels, batch_size, index):

    # controllo che il batch_size non sia più grande dell'intero array
    if batch_size < total:

        #controllo che uno dei due estremi non vada oltre la grandezza dell'array, in caso contrario significa che sono arrivato alla fine
        if (batch_size*index > total or batch_size*(index+1) > total):
            # ci sono due casi in cui si va oltre la grandezza totale dell'array:
            # 1) ho già percorso l'array almeno una volta ma voglio continuare a percorrerlo per continuare l'allenamento: batch_size*index continuerà ad aumentare anche dopo aver percorso l'array per la prima volta
            #    quindi uso %total per calcolare ogni volta gli indici nel range [0,total];
            # 2) devo prelevare l'ultimo batch ma non sono rimasti elementi sufficienti ovvero sono rimasti meno di batch_size elementi: prendo gli elementi disponibili e quelli mancanti li prendo ripercorrendo
            #    l'array dall'inizio.
            first_idx = (index*batch_size) % total
            second_index = ((index+1)*batch_size) % total

            # caso 1)
            if (first_idx < second_index):
                batch_xs = images[first_idx:second_index]
                batch_ys = labels[first_idx:second_index]

            # caso 2)
            else:
                batch_xs = images[first_idx:total]
                batch_xs = np.concatenate((batch_xs, images[0:second_index]), axis=0)
                batch_ys = labels[first_idx:total]
                batch_ys = np.concatenate((batch_ys, labels[0:second_index]), axis=0)

        # questo è il caso in cui si ricade la prima volta che si percorre l'array infatti non ho bisogno di usare %total perché tanto so per certo che batch_size*index
        # e/o batch_size*(index+1) non saranno mai maggiori di total (altrimenti si entrerebe nell'if e non nell'else)
        else:
            batch_xs = images[batch_size*index:batch_size*(index+1)]
            batch_ys = labels[batch_size*index:batch_size*(index+1)]

    # se batch_size è maggiore dell'intero array allora il mio batch sarà semplicemente composto dall'array stesso
    else:
        batch_xs = images
        batch_ys = labels

    return batch_xs, batch_ys

""" MAIN """
os.system('clear')
files_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'
img_size_w = 48
img_size_h = 1000
n_input = 48000 # w*h
mode = 'TRAIN'

total, images_arr, label_arr = get_images(
    files_path=files_path,
    img_size_w=img_size_w,
    img_size_h=img_size_h,
    n_input=n_input,
    mode=mode)


label_log = open(files_path + '/label_log.txt', 'w')
label_log = open(files_path + '/label_log.txt', 'a')
label_log.write('Labels_pre:\n\n' + str(label_arr))

images_log = open(files_path + '/images_log.txt', 'w')
images_log = open(files_path + '/images_log.txt', 'a')
images_log.write('Images_pre:\n\n' + str(images_arr))

print('Randomizzo i dataset...')
indices = np.arange(images_arr.shape[0])
np.random.shuffle(indices)
images_arr = images_arr[indices]
label_arr = label_arr[indices]

label_log.write('\n\n\nLabels_post:\n\n' + str(label_arr))
label_log.close()

#images_log.write('\n\nImages_post:\n\n' + str(images_arr[0:5]))
images_log.write('\n\nImages_post:\n\n' + str(np.array2string(images_arr[0:5], threshold=np.nan, max_line_width=np.nan)))
images_log.close()

print('\nreturn get_images:')
print(' - total: ' + str(total))
print(' - image_arr: ' + str(images_arr.shape))
print(' - label_arr: ' + str(label_arr.shape))

print('Estraggo i batch...')
print('\n')
for i in range(0,5):
    # creo un file per il log
    log2 = open(files_path + 'log' + str(i) + '.txt', 'w')
    log2 = open(files_path + 'log' + str(i) + '.txt', 'a')
    log2.write('Matrice ' + str(i) + '\n')

    batch_xs, batch_ys = next_batch(
        total = total, 
        images = images_arr,
        labels = label_arr,
        batch_size = 2,
        index = i
        )
    
    print('return next_batch_ ' + str(i) + ':')
    print(' - batch_xs: ' + str(batch_xs.shape))
    print(' - batch_ys: ' + str(batch_ys.shape) + '\n')
        
    # salvo il batch appena estratto su un file
    log2.write(str(np.array2string(batch_xs, threshold=np.nan, max_line_width=np.nan)))



"""
IMAGE RESIZE

Image.resize(size, resample=0)
    Returns a resized copy of this image.

    Parameters:
    size – The requested size in pixels, as a 2-tuple: (width, height).
    resample – An optional resampling filter. This can be one of PIL.Image.NEAREST(use nearest neighbour), PIL.Image.BILINEAR(linear interpolation), PIL.Image.BICUBIC(cubic spline interpolation), or PIL.Image.LANCZOS(a high-quality downsampling filter). If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
    Returns:
    An Image object.
"""
