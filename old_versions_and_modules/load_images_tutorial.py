from __future__ import absolute_import, division, print_function
import os
import pathlib
import random
import os
import IPython.display as display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

tf.enable_eager_execution()
tf.__version__
os.system('clear')
AUTOTUNE = tf.contrib.data.AUTOTUNE

data_root = pathlib.Path('/Users/matteo/Desktop/DATASET_X/SELECTION/TRAIN_IMG')
print('[LOG:data_root]: ' + str(data_root))

all_image_paths = []
all_image_labels = []
for item in data_root.iterdir():
    item_tmp = str(item)
    if 'selection.png' in item_tmp:
        all_image_paths.append(str(item))
        all_image_labels.append(0)
        print('[LOG:item]: ' + str(item))

print('[LOG:all_images_paths]:')
print(all_image_paths)
print('\n')
image_count = len(all_image_paths)
print('[LOG:image_count]: ' + str(image_count))

#print(all_image_paths[:10])

label_names = ['selection', 'neutral']

label_to_index = dict((name, index) for index, name in enumerate(label_names))
print('[LOG:all_image_labels]: ' + str(all_image_labels))

#print("First 10 labels indices: ", all_image_labels[:10])

img_path = all_image_paths[0]
# img_path

img_raw = tf.read_file(img_path)
print('[LOG:type(img_raw)]: ' + str(type(img_raw)))

# print('\n[LOG:repr]: ' + str(repr(img_raw)[:100]+"..."))

img_tensor = tf.image.decode_png(
    contents=img_raw,
    channels=1
)
log = open('/Users/matteo/Desktop/log.txt', 'w')
log = open('/Users/matteo/Desktop/log.txt', 'a')
log.write(str(img_tensor))
log.close()

print('\n[LOG:img_tensor.shape]: ' + str(img_tensor.shape))
print('[LOG:img_tensor.dtype]: ' + str(img_tensor.dtype))
print('\n')

# img_final = tf.image.resize_images(img_tensor, [1000, 48])
# #img_final = img_final/255.0
# print(img_final.shape)
print(img_tensor.numpy().min())
print(img_tensor.numpy().max())


def load_and_decode_image(path):
    image = tf.read_file(path)
    
    image = tf.image.decode_png(
        contents=image,
        channels=3
    )

    # image = tf.image.resize_images(image, [16, 16])
    
    return image


image_path = all_image_paths[0]
label = all_image_labels[0]

image = load_and_decode_image(image_path)
print('\n[LOG:image.shape]: ' + str(image.shape))
print('[LOG:image.dtype]: ' + str(image.dtype))

# plt.imshow(image)
# plt.grid(False)
# plt.xlabel(' 1.selection-png'.encode('utf-8'))
# plt.title(label_names[label].title())
# plt.show()

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print('[LOG:path_ds]:' + str(path_ds))

image_ds = path_ds.map(load_and_decode_image, num_parallel_calls=AUTOTUNE)

print('[LOG:image_ds]:' + str(image_ds))


# plt.figure(figsize=(8, 8))
# for n, image in enumerate(image_ds.take(4)):
#     plt.subplot(2, 2, n+1)
#     plt.imshow(image)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(' 1.selection-png'.encode('utf-8'))
# plt.show()
