from skimage.io import imsave
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from image_generator_modules import matrix_to_image_fn
from image_generator_modules import dir_restore

converter_fn = matrix_to_image_fn.converter_fn
dir_restore = dir_restore.dir_restore

os.system('clear')

MANUAL_MODE = True

# MATRICI DA GENERARE
n_mat = 500                      # numerop di matrici da generare
dimension = 50                  # dimensione delle matrici da generare
delta = 40.0                     # parametro delta per la creazione delle matrici
name_mat = 'image_generator'    # nome da assegnare alle matrici una volta generate

if MANUAL_MODE == True:
    n_mat = int(input('Numero di matrici da generare: '))
    dimension = int(input('Dimensione delle singole matrici: '))
    delta = int(input('Valore del parametro Delta: '))

    while(1):
        square = input('Vuoi che la matrice sia quadrata? [S/N]: ')

        if square == 'S':
            break
        elif square == 'N':
            break
  
    print('\n')

# NOME DEI FILE CONVERTITI IN .PNG
name_img = 'data'

path = str(os.getcwd()) + '/dataset/'
print(path)

dir_restore(path=path)

print('\nGenero i dataset e ne estraggo le matrici')
# GENERO LE MATRICI IN FORMATO .TXT
for i in tqdm(range(1, n_mat+1)):


    # Uso ms per generare le matrici in msdir/dataset/1_original_dataset/
    if square == 'N':
        os.system('cd msdir ; ./ms ' + str(dimension) + ' 1 -t ' + str(delta) + ' > ' + path + '/1_original_dataset/' + '1_' + name_mat + '_' + str(i) + '.txt')
        jump = 7
    elif square == 'S':
        os.system('cd msdir ; ./ms ' + str(dimension) + ' 1 -t ' + str(delta) + ' -s ' + str(dimension) + ' > ' + path + '/1_original_dataset/' + '1_' + name_mat + '_' + str(i) + '.txt')
        jump = 8
    
    row = 1

    # Apro l' n-esimo file .txt creato da ms
    with open(path + '1_original_dataset/' + '1_' + name_mat + '_' + str(i) + '.txt', "r") as fileobj:
        
        # Apro un file dove scrivere la matrice risistemata
        dest = open(path + '2_reshaped_dataset/' + '2_' + name_mat + '_' + str(i) + '_reshaped.txt', 'w')
        for line in fileobj: 
            if row < jump:
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
                file_name = path + '3_image_dataset/' + '3_' + name_mat + '_' + str(i) + '.png',
                name = name_mat + '_' + str(i) + '.png'
                ))
    log.write(line + '\n')


