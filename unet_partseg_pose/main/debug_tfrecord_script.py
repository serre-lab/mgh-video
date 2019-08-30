import random
import tensorflow as tf
import os
import pickle
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from central_reservoir.models import unet
from central_reservoir.augmentations import preprocessing_volume

from absl import flags
from absl import app

# Get train tfrecords
tfrecord_file = '/media/data_cifs/lakshmi/bmvc2018/convolutional-pose-machines-tensorflow/cpm_worm_train_combined.tfrecords'
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_folder_name',
    default='unet_bootstrappedv1_adam1e-4_newl2',
    help='To mention the model path')

flags.DEFINE_integer(
    'input_size',
    default=448,
    help='Input size to the model')

flags.DEFINE_integer(
    'num_classes',
    default=10,
    help='num joints + background')

flags.DEFINE_integer(
    'batch_size',
    default=1,
    help='batch size (choose according to memory availability')

flags.DEFINE_integer(
    'num_iters',
    default=100,
    help='Number of samples to view')

def fliplr(image,label):
    return tf.image.flip_left_right(image), tf.image.flip_left_right(label)

def flipud(image,label):
    return tf.image.flip_up_down(image), tf.image.flip_up_down(label)

def adjust_bright(image,label):
    return tf.image.random_brightness(image, max_delta=0.2), label

def adjust_contrast(image,label):
    return tf.image.random_contrast(image, lower=0.9, upper=1.1), label

def random_translate(image, 
		label,
		border=20):	
    # create a pad tensor of width = border. Rank will be 3, since
    # both image and labels are of rank 3
    paddings = tf.constant([[border,border],[border,border],[0,0]])
    # mimic the image/label along these dimensions
    padded_image = tf.pad(image,paddings,"SYMMETRIC")
    padded_label = tf.pad(label,paddings,"SYMMETRIC")
    # choose an offset height and width to crop from
    offsets = tf.random_uniform([2],minval=0, maxval=2*border, dtype=tf.int32)
    # crop the image from this offset
    image = tf.image.crop_to_bounding_box(padded_image, offsets[0], offsets[1], FLAGS.input_size, FLAGS.input_size)
    label = tf.image.crop_to_bounding_box(padded_label, offsets[0], offsets[1], FLAGS.input_size, FLAGS.input_size)
    return image, label

def no_aug(image,label):
    return image, label

def augment_patch(patch, gt):
    options = {
        0: fliplr,
        1: flipud,
        2: no_aug,
        3: adjust_bright,
	4: adjust_contrast,
	5: random_translate
    }
    aug_method = np.random.randint(5, size=1)
    aug_method = 5
    #return options[aug_method[0]](patch, gt)
    return random_translate(patch,gt)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, ser = reader.read(filename_queue)

    keys_to_features = {
        'image': tf.FixedLenFeature([], tf.string),
        'heatmaps': tf.FixedLenFeature([], tf.string) }

    parsed = tf.parse_single_example(
        ser,
        features=keys_to_features)

    image = tf.decode_raw(
        parsed['image'],
        tf.uint8)

    heatmaps = tf.decode_raw(
        parsed['heatmaps'],
        tf.float32)

    image = tf.reshape(image, [FLAGS.input_size, FLAGS.input_size, 3])
    heatmaps = tf.reshape(heatmaps, [FLAGS.input_size,FLAGS.input_size,FLAGS.num_classes])
    image, heatmaps = augment_patch(image, heatmaps)

    image = tf.cast(image, tf.float32) / 255.

    #heatmaps = tf.argmax(heatmaps,axis=-1)
    #heatmaps = tf.one_hot(heatmaps, FLAGS.num_classes)

    image, heatmaps = tf.train.shuffle_batch([image, heatmaps],
        seed=1234,
        batch_size=FLAGS.batch_size,
        capacity=100,
        num_threads=1,
        min_after_dequeue=50)

    '''
    videos, labels, heights, widths, nz = tf.train.shuffle_batch(
        [image, label, height, width, non_zero],
        seed=1234,
        batch_size=1,
        capacity=100,
        num_threads=1,
        min_after_dequeue=50)
    '''

    return image, heatmaps

def main(unused_argv):

    with tf.Session().as_default() as sess:
        # read from input tfrecord
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file],
            num_epochs=None)
        image, heatmaps = read_and_decode(filename_queue)
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        #try:
        for training_iter in range(FLAGS.num_iters):
            if coord.should_stop():
                break
            image_np, heatmaps_np = sess.run([image, heatmaps])
	    import ipdb; ipdb.set_trace()
            plt.imshow(image_np.squeeze()); plt.show(block=False); plt.figure(); plt.imshow(np.amax(heatmaps_np.squeeze()[:,:,:-1],axis=-1)); plt.show();
        coord.request_stop()
        #    print('Done...')
        #except:
        #    print('Some problem, good luck!')

if __name__ == '__main__':
    app.run(main)
