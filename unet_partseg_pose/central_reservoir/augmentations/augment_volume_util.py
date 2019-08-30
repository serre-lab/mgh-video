import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

'''
TODO : for all these methods
if label is a scalar, then leave it untouched
if it is an image (ex: pixel-wise segmentation) then crop it accordingly also
'''

'''
randomly crop a region uniformly through the volume
sample function call: random_crop_volume(volume,label,[64,224,224,3])
'''
def random_crop_volume(volume=None, 
                label=None, 
                target_dims=None, 
                seed=None):
    """Wrapper for random cropping.
    Modify tf random crop for 4D.
    """      

    vol_size = volume.get_shape().as_list()
    # get the new frame size specified
    tsize = target_dims[1:3]
    name = 'rand_crop'

    # Concat volume and label into a single volume for cropping
    #combined_volume = tf.concat([volume, label], axis=-1)
    #comb_size = combined_volume.get_shape().as_list()
    #crop_size = [comb_size[0]] + tsize + [comb_size[-1]]
    crop_size = [vol_size[0]] + tsize + [vol_size[-1]]

    with ops.name_scope(
            name, 'random_crop', [volume, crop_size]) as name:

        volume = ops.convert_to_tensor(volume, name='value')
        crop_size = ops.convert_to_tensor(
                crop_size, dtype=tf.int32, name='size')
        vol_shape = array_ops.shape(volume)

        # just making sure that the requested crop is not larger than
        # the exisiting size of the volume
        control_flow_ops.Assert(
                math_ops.reduce_all(vol_shape >= crop_size),
                ['Need vol_shape >= vol_size, got ', vol_shape, crop_size],
                summarize=1000)
        
        limit = vol_shape - crop_size + 1
        offset = tf.random_uniform(
                array_ops.shape(vol_shape),
                dtype=crop_size.dtype,
                maxval=crop_size.dtype.max,
                seed=seed) % limit
        cropped_volume = array_ops.slice(
                volume, offset, crop_size, name=name)
    #cropped_volume = cropped_combined[:, :, :, :vol_size[-1]]
    #cropped_label = cropped_combined[:, :, :, vol_size[-1]:]

    if not label is None:
        cropped_label = label
        return cropped_volume, cropped_label
    else:
        return cropped_volume


# currently valid directions: lr and ud
def flip(volume=None, 
        label=None, 
        direction=None, 
        rtn=False):

    """Flip volume and label pair in a direction."""
    if rtn:
        if not label is None:
            return volume, label
        else:
            return volume

    # Concat volume and label into a single volume for flipping
    vol_size = volume.get_shape().as_list()
    #combined_volume = tf.concat([volume, label], axis=-1)
    if direction == 'lr':
        combined_volume = array_ops.reverse(volume, [2])
    elif direction == 'ud':
        combined_volume = array_ops.reverse(volume, [1])
    else:
        raise NotImplementedError('Direction: %s not implemented.')
    flipped_volume = combined_volume[:, :, :, :vol_size[-1]]
    #flipped_label = combined_volume[:, :, :, vol_size[-1]:]

    if not label is None:
        flipped_label = label
        return flipped_volume, flipped_label
    else:
        return flipped_volume

# make a random flip based on a uniform distribution
# on an average, half the times this doesn't flip!
# sample function call: random_flip(volume,label, 'lr', threshold=0.5)
def random_flip(volume=None, 
            label=None, 
            direction=None, 
            threshold=0.5, 
            seed=None):
    """Random flip volume and label pair in a direction."""

    if label is None:
        flipped_volume = tf.cond(
                    tf.greater(
                        tf.random_uniform([], 0, 1.0, seed=seed), threshold),
                        lambda: flip(volume=volume, direction=direction),
                        lambda: flip(volume=volume, direction=direction, rtn=True))

    if not label is None:
        # to be implemented
        flipped_volume = volume
        flipped_label = label      
        return flipped_volume, flipped_label
    else:
        return flipped_volume
