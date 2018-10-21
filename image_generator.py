from skimage.io import imsave
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matrix_to_image_2
converter_fn = matrix_to_image_2.converter_fn

os.system('clear')

# MATRICI DA GENERARE
n_mat = 10                      # numerop di matrici da generare
dimension = 100                  # dimensione delle matrici da generare
delta = 27.0                     # parametro delta per la creazione delle matrici
name_mat = 'image_generator'    # nome da assegnare alle matrici una volta generate

# NOME DEI FILE CONVERTITI IN .PNG
name_img = 'data'
path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/msdir/dataset/'

# GENERO LE MATRICI IN FORMATO .TXT
for i in range(1, n_mat+1):
    os.system('cd msdir ; ./ms ' + str(dimension) + ' 1 -t ' + str(delta) +' > dataset/' + '1_' + name_mat + '_' + str(i) + '.txt')
    row = 1
    with open(path + '1_' + name_mat + '_' + str(i) + '.txt', "r") as fileobj:
        dest = open(path + '2_' + name_mat + '_' + str(i) + '_reshaped.txt', 'w')
        for line in fileobj: 
            if row < 7:
                row = row+1
            elif len(line) > 5:
                dest.write(line)
                row = row + 1

for i in range(1, n_mat+1):
    print(converter_fn(
                path = path + '2_' + name_mat + '_' + str(i) + '_reshaped.txt',
                file_name = path + '3_' + name_mat + '_' + str(i) + '.png'
                ))


