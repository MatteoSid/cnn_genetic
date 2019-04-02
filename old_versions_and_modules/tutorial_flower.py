from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import os
import pathlib
import random
import os
import IPython.display as display
import tensorflow as tf
import numpy as np

np.set_printoptions(threshold=np.nan)
tf.enable_eager_execution()
tf.__version__
os.system('clear')
# AUTOTUNE = tf.data.experimental.AUTOTUNE
AUTOTUNE = tf.contrib.data.AUTOTUNE

log = open('/Users/matteo/Desktop/log.txt', 'w')
log = open('/Users/matteo/Desktop/log.txt', 'a')

data_root_orig = tf.keras.utils.get_file('flower_photos',
                                         'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         untar=True)

data_root = pathlib.Path(data_root_orig)
print('[LOG:data_root]: ' + str(data_root))

for item in data_root.iterdir():
  print('[LOG:item]: ' + str(item))


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

# print('[LOG:type(all_image_paths)]: ' + str(type(all_image_paths)))
# print('[LOG:all_image_paths]: ' + str(all_image_paths))

image_count = len(all_image_paths)

all_image_paths[:10]


attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


# print('\n')
# for n in range(3):
#     image_path = random.choice(all_image_paths)
#     display.display(display.Image(image_path))
#     print('[LOG:caption_image]: ' + str(caption_image(image_path)))
#     print()
# print('\n')

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print('\n[LOG:type(label_names)]: ' + str(type(label_names)))
print('[LOG:label_names]:\n' + str(label_names))

label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
# print(all_image_labels)

print("First 10 labels indices: ", all_image_labels[:10])
print('\n')

img_path = all_image_paths[0]

img_raw = tf.read_file(img_path)
print('\n[LOG:repr]: ' + str(repr(img_raw)[:100]+"..."))

img_tensor = tf.image.decode_image(img_raw)

print('\n[LOG:img_tensor.shape]: ' + str(img_tensor.shape))
print('[LOG:img_tensor.dtype]: ' + str(img_tensor.dtype))
print('\n\n')

img_final = tf.image.resize_images(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


def preprocess_image(image):
    log.write('\n[LOG:preprocess_image]: ' + str(image))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    log.write('\n[LOG:load_and_preprocess_image]: ' + str(path))
    image = tf.read_file(path)
    return preprocess_image(image)


image_path = all_image_paths[0]
label = all_image_labels[0]

image = load_and_preprocess_image(image_path)
print('\n[LOG:image.shape]: ' + str(image.shape))
print('[LOG:image.dtype]: ' + str(image.dtype))

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path).encode('utf-8'))
plt.title(label_names[label].title())
# plt.show()

log.write('\n\n\n\n\n')
all_image_paths = tf.convert_to_tensor(all_image_paths)
log.write(str(tf.transpose(all_image_paths)))
print('\n[LOG:all_image_paths.shape]: ' + str(all_image_paths.shape))
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print('shape: ' + str(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)

image_ds = path_ds.map(
    load_and_preprocess_image, 
    num_parallel_calls=AUTOTUNE
    )
print('[LOG:image_ds]:' + str(image_ds))
# print('[LOG:image_ds.shape]:' + str(image_ds.shape))
print('[LOG:type(image_ds)]:' + str(type(image_ds)))

# plt.figure(figsize=(8, 8))
# for n, image in enumerate(image_ds.take(4)):
#     print('[LOG:n, image]: ' + str(n) + ', ' + str(image))
#     plt.subplot(2, 2, n+1)
#     plt.imshow(image)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(caption_image(all_image_paths[n]))
# plt.show()

