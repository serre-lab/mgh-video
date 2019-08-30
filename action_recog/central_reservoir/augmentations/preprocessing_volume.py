import tensorflow as tf

import augment_volume_util as avu

#TODO: 3D label stuff for seg, target_image_size is 1 int, can be list for !square images
def preprocess_for_train(volume,
                        num_frames,
                        height,
                        width,
                        target_image_size,
                        list_of_augmentations=[]):

    '''Preprocessing volumes during training.
    Args:
        volume: '4D Tensor' [depth, height, width, channels]
        target_image_size: 'Integer' to specify input size
            required by the model
        list_of_augmentations: 'List' of strings specifying
            augmentations to be done
    Returns:
        Augmented '4D Tensor of same dtype as :volume
    '''

    get_shape = volume.get_shape().as_list()
    assert len(get_shape) == 4, 'Input shape length should be\
        4. Found %d' %len(get_shape)
    if len(list_of_augmentations) == 0:
        print('No augmentations mentioned, function will return\
            volume unchanged')
    
    ##### Random cropping
    target_dims = [
        num_frames,
        target_image_size,
        target_image_size,
        3]

    if 'random_crop' in list_of_augmentations:
        volume = avu.random_crop_volume(
            volume=volume,
            target_dims=target_dims)


    '''
    ##### Random flipping
    if 'random_flip' in list_of_augmentations:
        volume = avu.random_flip(
            volume=volume,
            direction='lr')
    '''

    ##### Brightness in range [0.0, 0.3)
    if 'random_brightness' in list_of_augmentations:
        volume = avu.apply_brightness(
            volume=volume)

    ##### Contrast in range [0.0, 0.3)
    if 'random_contrast' in list_of_augmentations:
        volume = avu.apply_contrast(
            volume=volume)

    return volume


def preprocess_for_eval(volume,
                    num_frames,
                    height,
                    width,
                    target_image_size,
                    list_of_augmentations=[]):

    
    ##### Crop center 224*224 patch
    #height, width = get_shape[1], get_shape[2]
    center_x = tf.cast(
        tf.divide(
            height
                if type(height) == int else height[0],
            2),
        tf.int32)

    center_y = tf.cast(
        tf.divide(
            width
                if type(width) == int else width[0],
            2),
        tf.int32)

    offset_height = tf.subtract(
        center_x,
        112)
    offset_width = tf.subtract(
        center_y,
        112)
    target_height, target_width = target_image_size,\
        target_image_size
    volume = tf.image.crop_to_bounding_box(
        volume,
        offset_height,
        offset_width,
        target_height,
        target_width)

    # for testing ucf-101. comment otherwise
    #volume = tf.slice(
    #    volume,
    #    [0, 0, 0, 0],
    #    [64, 224, 224, 3])
    
    return volume


def preprocess_volume(volume,
                    num_frames,
                    height,
                    width,
                    is_training=False,
                    target_image_size=224,
                    use_bfloat16=False,
                    list_of_augmentations=[]):

    '''Preprocess the given image.

    Args:
        1. volume: Tensor representing an uint\
            volume of arbitrary size
        2. height: Tensor representing the original\
            height of the volume
        3. width: Tensor representing the original\
            width of the volume
        4. is_training: bool for whether the\
            preprocessing is for training
        5. target_image_size: int for representing input\
            size to the model
        6. num_frames: int for representing the\
            number of frames in a volume
        7. use_bfloat16: bool for whether to use\
            bfloat16
        8. list_of_augmentations: Specify augmentation\
            schemes
    
    Returns:
        A preprossed image Tensor with value range\
            of [-1, 1].
    '''
    
    # Get back actual volume shape
    if is_training:
        volume = tf.reshape(
            volume,
            [
                num_frames,
                height,
                width,
                3])
    else:
        volume = tf.reshape(
            volume,
            [
                num_frames,
                height,
                width,
                3])

    if is_training:
        func = preprocess_for_train
    else:
        func = preprocess_for_eval

    volume = func(
        volume,
        num_frames,
        height,
        width,
        target_image_size,
        list_of_augmentations)
   
    ##### Cast volume to float32
    volume = tf.cast(
        volume,
        tf.float32)

    ##### I3d takes input in range [-1, 1]
    volume = tf.subtract(
        tf.divide(
            volume,
            tf.constant(
                127.5,
                dtype=tf.float32)),
        tf.constant(
            1.,
            dtype=tf.float32))

    if use_bfloat16:
        ##### Conversion to bfloat16
        volume = tf.image.convert_image_dtype(
            volume,
            dtype=tf.bfloat16)

    return volume
