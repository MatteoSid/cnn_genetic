from skimage.io import imsave
import os
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

os.system('clear')

path = "/Users/matteo/Desktop/matrix_to_image"

#os.mkdir(path)


# array = []
# count = 0
# for line in file:
#     stringa = file.readline(44)
#     array.append(int(stringa))
#     count = count +1
#     #file.readline()
#     #print(str(count) + ':\t' + stringa)

# print(array.dtype)
innerarray=[]
array=[]

w=0
h=0
with open("/Users/matteo/Desktop/matrix_to_image/matrix.txt","r") as fileobj:
    for line in fileobj:  
       w=0
       for ch in line:
           if ch!='\n':
            innerarray.append(float(ch))
            w=w+1
       h=h+1
       array.append(innerarray)
       innerarray=[]

print("H: ", str(h), " W: ", str(w)) 

#for x in array:
 #   print(x)
array=np.asarray(array)
print(type(array))
#A = rand(5,5)
# figure(1)
# plt.imshow(array)
#plt.show()

fig = plt.figure(frameon=False, figsize=(w,h))



ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.style.use('grayscale')


ax.imshow(array, aspect='auto')
#ax.show()
#fig.savefig('/Users/matteo/Desktop/matrix_to_image/matrixtoimage.png', bbox_inches='tight', pad_inches=0)
fig.savefig('/Users/matteo/Desktop/matrix_to_image/matrixtoimage.png' ,dpi=1)

#plt.savefig('/Users/matteo/Desktop/matrix_to_image/matrixtoimage.png')
