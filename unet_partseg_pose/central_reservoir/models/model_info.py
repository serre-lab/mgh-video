import i3d
import unet

import argparse

import tensorflow as tf

ID_MODELNAME = {
    1: 'I3d',
    2: 'Unet'}

inputs_2d = tf.ones(
    [1, 224, 224, 3],
    dtype=tf.float32)

inputs_3d = tf.ones(
    [1, 64, 224, 224, 3],
    dtype=tf.float32)

parser = argparse.ArgumentParser(
    description='params of running the experiment')
parser.add_argument(
    '--model_id',
    type=int,
    default=1,
    help='To specify which model to test run')

args = parser.parse_args()
model_id = args.model_id

if model_id == 1:
    network = i3d.InceptionI3d()
    inputs = inputs_3d

elif model_id == 2:
    network = unet.UNET()
    inputs = inputs_2d

else:
    print('Model id {} not valid'.format(model_id))

print('Model : {}'.format(ID_MODELNAME[model_id]))
logits, end_points = network(
    inputs,
    is_training=True)
