from skimage.io import imsave
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matrix_to_image_fn
import dir_restore
import time
from tqdm import tqdm
converter_fn = matrix_to_image_fn.converter_fn
dir_restore = dir_restore.dir_restore

os.system('clear')

# MATRICI DA GENERARE
n_mat = 50                      # numerop di matrici da generare
dimension = 100                  # dimensione delle matrici da generare
delta = 40.0                     # parametro delta per la creazione delle matrici
name_mat = 'image_generator'    # nome da assegnare alle matrici una volta generate

# NOME DEI FILE CONVERTITI IN .PNG
name_img = 'data'
path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/dataset/'

dir_restore(path=path)

print('\nGenero i dataset e ne estraggo le matrici')
# GENERO LE MATRICI IN FORMATO .TXT
for i in tqdm(range(1, n_mat+1)):


    # Uso ms per generare le matrici in msdir/dataset/1_original_dataset/
    os.system('cd msdir ; ./ms ' + str(dimension) + ' 1 -t ' + str(delta) +' > ' + path + '/1_original_dataset/' + '1_' + name_mat + '_' + str(i) + '.txt')
    row = 1

    # Apro l' n-esimo file .txt creato da ms
    with open(path + '1_original_dataset/' + '1_' + name_mat + '_' + str(i) + '.txt', "r") as fileobj:
        
        # Apro un file dove scrivere la matrice risistemata
        dest = open(path + '2_reshaped_dataset/' + '2_' + name_mat + '_' + str(i) + '_reshaped.txt', 'w')
        for line in fileobj: 
            if row < 7:
                row = row+1
            elif len(line) > 5:
                dest.write(line)
                row = row + 1

log = open(path + '3_image_dataset/image_log.txt', 'w')
log = open(path + '3_image_dataset/image_log.txt', 'a')
print('\nTrasformo le matrici in immagini .png')
for i in tqdm(range(1, n_mat+1)):
    line = (converter_fn(
                path = path + '2_reshaped_dataset/' + '2_' + name_mat + '_' + str(i) + '_reshaped.txt',
                file_name = path + '3_image_dataset/' + '3_' + name_mat + '_' + str(i) + '.png'
                ))
    log.write(line + '\n')


