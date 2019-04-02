import os
import tensorflow as tf
from load_dataset import get_images
#get_images = load_dataset.get_images

os.system('clear')
files_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'

# step 1
filenames = tf.constant(['1.selection.png', '2.selection.png','3.selection.png', '4.selection.png', '5.selection.png'])
labels=[]
for i in range(0, 5):
    labels.append([0,1])
labels = tf.constant(labels)



print(filenames.shape)
print(labels.shape)

print('\n')
print(filenames)
print(labels)

print('\n')

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))


# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    print(image_string)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    print(image_decoded)
    image = tf.cast(image_decoded, tf.float32)
    print(image)
    return image, label


dataset = dataset.map(_parse_function)
dataset = dataset.batch(2)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()



