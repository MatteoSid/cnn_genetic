import os
import tensorflow as tf
from load_dataset import get_images
#get_images = load_dataset.get_images

os.system('clear')
files_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'

tmp_images = []
tmp_layer = []

n, images, layers = get_images(
    files_path=files_path,
    img_size_w=48,
    img_size_h=1000,
    mode='TRAIN',
    randomize=False
)

# for files in os.listdir(files_path + 'SELECTION/TRAIN_IMG'):
#     file_name = str(files)
#     if file_name.endswith('.png'):
#         tmp_images.append(files)
#         tmp_layer.append([0,1])


print(images.shape)
print(layers.shape)

images = tf.constant(images)
layers = tf.constant(layers)

print('\n')
print(images)
print(layers)

print('\n')

dataset = tf.data.Dataset.from_tensor_slices((images, layers))

print(tf.read_file(dataset))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


# dataset = dataset.map(_parse_function)
# dataset = dataset.batch(2)

# # step 4: create iterator and final input tensor
# iterator = dataset.make_one_shot_iterator()
# images, labels = iterator.get_next()



