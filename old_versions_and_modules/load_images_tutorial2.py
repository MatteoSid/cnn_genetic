from __future__ import absolute_import, division, print_function
import os
import pathlib
import IPython.display as display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

tf.enable_eager_execution()
tf.__version__
os.system('clear')
AUTOTUNE = tf.contrib.data.AUTOTUNE

#### some tries for the SELECTION dataset ####

data_root = pathlib.Path('/Users/matteo/Desktop/DATASET_X/SELECTION/TRAIN_IMG')

all_image_paths = []
all_image_labels = []
for item in data_root.iterdir():
    item_tmp = str(item)
    if 'selection.png' in item_tmp:
        all_image_paths.append(str(item))
        all_image_labels.append(0)

image_count = len(all_image_paths)
label_names = ['selection', 'neutral']
label_to_index = dict((name, index) for index, name in enumerate(label_names))
img_path = all_image_paths[0]
img_raw = tf.read_file(img_path)

img_tensor = tf.image.decode_png(
    contents=img_raw,
    channels=1
)
print(img_tensor.numpy().min())
print(img_tensor.numpy().max())
#### it works fine till here ####

#### trying to make a function ####
#### problems from here ####

def load_and_decode_image(path):
    print('[LOG:load_and_decode_image]: ' + str(path))
    image = tf.read_file(path)
    
    image = tf.image.decode_png(
        contents=image,
        channels=3
    )
    
    return image


image_path = all_image_paths[0]
label = all_image_labels[0]

image = load_and_decode_image(image_path)
print('[LOG:image.shape]: ' + str(image.shape))

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print('[LOG:path_ds]:' + str(path_ds))



image_ds = path_ds.map(load_and_decode_image, num_parallel_calls=AUTOTUNE)
plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    print('[LOG:n, image]: ' + str(n) + ', ' + str(image))
    plt.subplot(2, 2, n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(' selection'.encode('utf-8'))
    plt.title(label_names[label].title())
plt.show()

# plt.imshow(image)
# plt.grid(False)
# plt.xlabel(' 1.selection-png'.encode('utf-8'))
# plt.title(label_names[label].title())
