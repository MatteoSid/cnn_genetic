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

    # Carico il SELECTION dataset [0,1]
    files = [f for f in listdir(selection_path) if isfile(join(selection_path, f))]
    print('\nCarico il SELECTION dataset:')
    for fl in tqdm(files):
        if fl != '.DS_Store':
            
            # carico le immagini convertendole in immagini ad un canale
            image = Image.open(selection_path + fl).convert(mode='L')  

            # Controllo che le immagini abbiano tutte la stessa dimensione, altrimenti le salto
            if str(image.size) == '(' + str(img_size_w) + ', ' + str(img_size_h) + ')':
                label_arr.append([0, 1])    # applico l'etichetta [0,1] a tutte le immagini SELECTION
                images_arr.append(np.array(image,dtype=int))
            
    # Carico il NEUTRAL dataset [1,0]
    files = [f for f in listdir(neutral_path) if isfile(join(neutral_path, f))] 
    print('\nCarico il NEUTRAL dataset:')
    for fl in tqdm(files):
        if fl != '.DS_Store':
            # carico le immagini convertendole in immagini ad un canale
            image = Image.open(neutral_path + fl).convert(mode='L')
            
            # Controllo che le immagini abbiano tutte la stessa dimensione, altrimenti le salto
            if str(image.size) == '(' + str(img_size_w) + ', ' + str(img_size_h) + ')':
                label_arr.append([1,0])         # applico l'etichetta [0,1] a tutte le immagini NEUTRAL
                images_arr.append(np.array(image, dtype=int))
        
    # trasformo le liste in array numpy
    images_arr = np.array(images_arr)
    label_arr = np.array(label_arr)

    print('\nGli array hanno dimensione:\n  -images_arr: ' + str(len(images_arr)) + '\n  -label_arr: ' + str(len(label_arr)))
    print('\nimages_array shape: ' + str(images_arr.shape))
    print('label_array shape: ' + str(label_arr.shape))
    
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

            first_idx = (index*batch_size) % total
            second_index = ((index+1)*batch_size) % total
            
            # ripercorro l'arrai per l'n-esima volta
            if (first_idx < second_index):
                batch_xs = images[first_idx:second_index]
                batch_ys = labels[first_idx:second_index]
            
            # uso gli elementi rimanenti e completo il batch con i primi elementi dell'array necessari per arrivare a batch_size elementi
            else:
                batch_xs = images[first_idx:total]
                batch_xs = np.concatenate((batch_xs, images[0:second_index]), axis=0)
                batch_ys = labels[first_idx:total]
                batch_ys = np.concatenate((batch_ys, labels[0:second_index]), axis=0)

        # prima volta che scorro l'array
        else:
            batch_xs = images[batch_size*index:batch_size*(index+1)]
            batch_ys = labels[batch_size*index:batch_size*(index+1)]

    # se batch_size è maggiore dell'intero array allora il mio batch sarà semplicemente composto dall'array stesso
    else:
        batch_xs = images
        batch_ys = labels

    return batch_xs, batch_ys


""""""""""""
""" MAIN """
""""""""""""
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

# randomizzo i due array allo stesso modo per non perdere la corrispondenza tra i due
print('Randomizzo i dataset...')
indices = np.arange(images_arr.shape[0])
np.random.shuffle(indices)
images_arr = images_arr[indices]
label_arr = label_arr[indices]

print('\nreturn get_images:')
print(' - total: ' + str(total))
print(' - image_arr: ' + str(images_arr.shape))
print(' - label_arr: ' + str(label_arr.shape))
print('\n')

print('Estraggo i batch...')
for i in range(0,5):
    batch_xs, batch_ys = next_batch(
        total = total, 
        images = images_arr,
        labels = label_arr,
        batch_size = 2,
        index = i
        )
    

