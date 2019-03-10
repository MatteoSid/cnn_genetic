from os import listdir
from PIL import Image, ImageChops
import random
from os.path import isfile, join
import numpy as np

# FUNZIONE PER REPERIRE SET DI IMMAGINI DAL FILE SYSTEM
# Funziona sia per il training set che per ogni altro set

def get_images(files_path, img_size_w, img_size_h, n_input):
    files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    images_arr = []
    label_arr=[]
    for fl in files:
        if fl[5:9]=='good':
            label_arr.append([0,1])
        else:
            label_arr.append([1,0])

        image = Image.open(files_path+fl)
        image = image.resize((img_size_w,img_size_h))

        images_arr.append(np.array(image,dtype=float))

    images_arr = np.array(images_arr)
    label_arr = np.array(label_arr)
    images_arr = images_arr.reshape(len(images_arr),n_input)
    return len(images_arr),images_arr,label_arr


# Image.resize(size, resample=0)
#     Returns a resized copy of this image.

#     Parameters:
#     size – The requested size in pixels, as a 2-tuple: (width, height).
#     resample – An optional resampling filter. This can be one of PIL.Image.NEAREST(use nearest neighbour), PIL.Image.BILINEAR(linear interpolation), PIL.Image.BICUBIC(cubic spline interpolation), or PIL.Image.LANCZOS(a high-quality downsampling filter). If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
#     Returns:
#     An Image object.

