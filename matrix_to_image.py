from skimage.io import imsave
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

os.system('clear')

path = "/Users/matteo/Desktop/cnn_mnist/matrix_to_image/"

innerarray=[]
array=[]

w=0
h=0
with open(path + "matrix.txt","r") as fileobj:
    for line in fileobj:  
       w=0
       for ch in line:
           if ch!='\n':
            innerarray.append(float(ch))
            w=w+1
       h=h+1
       array.append(innerarray)
       innerarray=[]

print('\nÈ stata creata un\'immagine di altezza=' + str(h) + ' e larghezza=' + str(w) + '.\n')
array=np.asarray(array)
#print(type(array))

fig = plt.figure(frameon=False, figsize=(w,h))

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.style.use('grayscale')

ax.imshow(array, aspect='auto')

fig.savefig(path + 'matrix_from_image.png' ,dpi=1)