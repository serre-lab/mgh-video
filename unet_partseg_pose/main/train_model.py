import random
import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from central_reservoir.models import unet
from central_reservoir.augmentations import preprocessing_volume

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_folder_name',
    default=None,
    help='To mention the model path')

flags.DEFINE_string(
    'common_path',
    default=None,
    help='Base directory')

flags.DEFINE_string(
    'TFR',
    default=None,
    help='Path to input train tfrecords')

flags.DEFINE_boolean(
    'weight_decay',
    default=False,
    help='Regularization y/n?')

flags.DEFINE_integer(
    'input_size',
    default=192,
    help='Input size to the model')

flags.DEFINE_integer(
    'num_classes',
    default=10,
    help='num joints + background')

flags.DEFINE_integer(
    'batch_size',
    default=8,
    help='batch size (choose according to memory availability')

flags.DEFINE_integer(
    'num_iters',
    default=200000,
    help='duration of training')

flags.DEFINE_boolean(
    'single_channel',
    default=False,
    help='just generic localization of joints')

def fliplr(image,label):
    return tf.image.flip_left_right(image), tf.image.flip_left_right(label)

def flipud(image,label):
    return tf.image.flip_up_down(image), tf.image.flip_up_down(label)

def adjust_bright(image,label):
    return tf.image.random_brightness(image, max_delta=0.2), label

def no_aug(image,label):
    return image, label

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


def augment_patch(patch, gt):
    #options = {
    #    0: fliplr,
    #    1: flipud,
    #    2: no_aug,
    #    3: adjust_bright,
	#4: adjust_contrast,
	#5: random_translate
    #}
    options = {
        0: no_aug,
        1: adjust_bright,
        2: adjust_contrast,
        3: random_translate
    }
    aug_method = np.random.randint(4, size=1)
    return options[aug_method[0]](patch, gt)

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
    image = image[..., ::-1]
    heatmaps = tf.reshape(heatmaps, [FLAGS.input_size,FLAGS.input_size,FLAGS.num_classes])
    image, heatmaps = augment_patch(image, heatmaps)

    image = tf.cast(image, tf.float32) / 255.

    image, heatmaps = tf.train.shuffle_batch([image, heatmaps],
        seed=1234,
        batch_size=FLAGS.batch_size,
        capacity=100,
        num_threads=1,
        min_after_dequeue=50)

    return image, heatmaps

def main(unused_argv):

    if FLAGS.common_path is None:
	raise ValueError('Please specify the root directory')
    else:
	common_path = FLAGS.common_path

    if FLAGS.model_folder_name is None:
	raise ValueError('Please specify a target model folder name')

    if FLAGS.TFR is None:
	raise ValueError('Train tfrecords path not specified')
    else:
	tfrecord_file = FLAGS.TFR

    summary_path = os.path.join(
        common_path,
        'summaries',
        FLAGS.model_folder_name)
    ckpt_path = os.path.join(
        common_path,
        'model_runs',
        FLAGS.model_folder_name)     

    with tf.Session().as_default() as sess:
        # read from input tfrecord
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file],
            num_epochs=None)
        image, heatmaps = read_and_decode(filename_queue)
        # create the model
        network = unet.UNET(
            use_batch_norm=True,
            use_cross_replica_batch_norm=False,
            num_classes=FLAGS.num_classes,
	    single_channel=FLAGS.single_channel)
        # get the readouts
        logits, end_points = network(
            inputs=image,
            is_training=True)

    #####
	# Loss function definition
	#####
	#import ipdb; ipdb.set_trace()
	if FLAGS.single_channel:
		#compressed_maps = tf.reduce_max(heatmaps[:,:,:,:-1],axis=-1,keepdims=True)

		# 3 joints case
		compressed_maps = tf.expand_dims(heatmaps[:,:,:,3] + heatmaps[:,:,:,7] + heatmaps[:,:,:,8],axis=-1)

		# 5 joints case
		#compressed_maps = tf.expand_dims(heatmaps[:,:,:,1] + heatmaps[:,:,:,3] + heatmaps[:,:,:,5] + heatmaps[:,:,:,7] + heatmaps[:,:,:,8],axis=-1)

		zero_entries = tf.equal(compressed_maps,0.)
		non_zero_entries = tf.not_equal(compressed_maps,0.)
		diff_entries = logits - compressed_maps
		loss_wd = 0.1 * tf.nn.l2_loss(tf.boolean_mask(diff_entries,zero_entries)) + 0.9 * tf.nn.l2_loss(tf.boolean_mask(diff_entries,non_zero_entries))
	else:
		loss_wd = tf.nn.l2_loss( logits - heatmaps, 'l2loss')
		#loss_wd = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(heatmaps,axis=-1),logits=logits))

	if FLAGS.weight_decay:
		wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
        	loss = loss_wd +(1e-6 * tf.add_n([tf.nn.l2_loss(x) for x in wd_l]))
	else:
		loss = loss_wd

        optimizer =  tf.train.AdamOptimizer(
                    learning_rate=1e-4)
        global_step = tf.train.get_global_step()        

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                        loss,
                        global_step)
        
        # for tensorboard
        tf.summary.scalar("l2_loss",loss)
        tf.summary.image("Inputs",image)
        tf.summary.image("Groundtruth", tf.reduce_max(heatmaps,axis=-1,keep_dims=True))
	if FLAGS.single_channel:
	        tf.summary.image("Predictions", logits)
	else:
	        #tf.summary.image("Predictions", tf.reduce_max(logits,axis=-1,keepdims=True))
		tf.summary.image("Predictions", tf.cast(tf.expand_dims(tf.argmax(logits,axis=-1),axis=-1),tf.float32) * 255. / FLAGS.num_classes)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary_path,sess.graph)

        # initialize the saver
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
	# restore weights
	ckpts = tf.train.latest_checkpoint(os.path.join(
						common_path,
						'model_runs',
						'unet_adam1e-4_preAllOpenPose'))
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess.run(init_op)
	#saver.restore(sess,ckpts)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        try:
            for training_iter in range(FLAGS.num_iters):
                if coord.should_stop():
                    break
                
		image_np, heatmaps_np, loss_np, _ = sess.run([image, heatmaps, loss, train_op])
                if (training_iter % 100) == 0:
                    print('step = {}, loss = {}'.format(training_iter,loss_np))
                print('step = {}, loss = {}'.format(training_iter,loss_np))
    
                # save checkpoints
                if (training_iter % 2000) == 0:
                    saver.save(sess,
                        os.path.join(ckpt_path,
                        'model-'+str(training_iter)+'.ckpt'),
                        global_step=training_iter)

                # tensorboard
                if (training_iter % 500) == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str,training_iter)
            coord.request_stop()
            print('Training done...')
        except:
            print('{} steps covered'.format(training_iter))


if __name__ == '__main__':
    app.run(main)
