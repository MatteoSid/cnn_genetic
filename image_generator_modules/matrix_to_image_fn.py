from skimage.io import imsave
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0


os.system('clear')

def converter_fn(path, file_name, name):
    # innerarray contiene le stringhe corrispondenti alle righe della matrice
    innerarray=[]

    # array è un array di array che contiene tutte le righe
    array=[]

    #altezza e larghezza della matrice trasformata
    w=0
    h=0

    with open(path, "r") as fileobj:
        for line in fileobj:  
            w=0
            for ch in line:
                if ch!='\n': # escludo i caratteri di andata a capo
                    innerarray.append(int(ch))
                    w=w+1
            h=h+1
            array.append(innerarray)
            innerarray=[]
    #log = 'Immagine salvata alla posizione: ' + path + 'image_from_' + file_name + '\nL\'immagine ha dimensione[h,w]=[' + str(h) + ', ' + str(w) + '].\n'
    log = name + ':\n' + 'L\'immagine ha dimensione[w,h]=[' + str(w) + ', ' + str(h) + '].\n'
    #print('L\'immagine ha dimensione[h,w]=[' + str(h) + ', ' + str(w) + '].\n')
    array=np.asarray(array)

    fig = plt.figure(frameon=False, figsize=(w,h))

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.style.use('grayscale')

    ax.imshow(array, aspect='auto')

    fig.savefig(file_name, dpi=1)

    return(log)