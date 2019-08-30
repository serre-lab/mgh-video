import tensorflow as tf

from augment_volume_util import *

#TODO: 3D label stuff for seg, target_image_size is 1 int, can be list for !square images
def preprocess_for_train(video,
                        num_frames,
                        height,
                        width,
                        target_image_size,
                        list_of_augmentations=[]):

    '''Preprocessing volumes during training.
    Args:
        video: '4D Tensor' [depth, height, width, channels]
        target_image_size: 'Integer' to specify input size
            required by the model
        list_of_augmentations: 'List' of strings specifying
            augmentations to be done
    Returns:
        Augmented '4D Tensor of same dtype as :video
    '''

    get_shape = video.get_shape().as_list()
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
        video = random_crop_volume(
            volume=video,
            target_dims=target_dims)


    ##### Random flipping
    if 'random_flip' in list_of_augmentations:
        video = random_flip(
            volume=video,
            direction='lr')

    return video


def preprocess_for_eval(video,
                    num_frames,
                    height,
                    width,
                    target_image_size):

    
    ##### Crop center 224*224 patch
    #height, width = get_shape[1], get_shape[2]
    center_x = tf.cast(
        tf.divide(
            height,
            2),
        tf.int32)

    center_y = tf.cast(
        tf.divide(
            width,
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
    video = tf.image.crop_to_bounding_box(
        video,
        offset_height,
        offset_width,
        target_height,
        target_width)
    # for testing ucf-101. comment otherwise
    #video = tf.slice(
    #    video,
    #    [0, 0, 0, 0],
    #    [64, 224, 224, 3])
    
    return video


def preprocess_volume(video,
                    num_frames,
                    height,
                    width,
                    is_training=False,
                    target_image_size=224,
                    use_bfloat16=False,
                    list_of_augmentations=[]):

    '''Preprocess the given image.

    Args:
        1. video: Tensor representing an uint\
            video of arbitrary size
        2. height: Tensor representing the original\
            height of the video
        3. width: Tensor representing the original\
            width of the video
        4. is_training: bool for whether the\
            preprocessing is for training
        5. target_image_size: int for representing input\
            size to the model
        6. num_frames: int for representing the\
            number of frames in a video
        7. use_bfloat16: bool for whether to use\
            bfloat16
        8. list_of_augmentations: Specify augmentation\
            schemes
    
    Returns:
        A preprossed image Tensor with value range\
            of [-1, 1].
    '''
    
    # Get back actual video shape
    if is_training:
        video = tf.reshape(
            video,
            [
                num_frames,
                height,
                width,
                3])
    else:
        video = tf.reshape(
            video,
            [
                num_frames[0],
                height[0],
                width[0],
                3])
            
    if is_training:
        video = preprocess_for_train(
            video,
            num_frames,
            height,
            width,
            target_image_size,
            list_of_augmentations)
    else:
        video = preprocess_for_eval(
            video,
            num_frames,
            height,
            width,
            target_image_size)

    ##### Cast video to float32
    video = tf.cast(
        video,
        tf.float32)

    ##### I3d takes input in range [-1, 1]
    video = tf.subtract(
        tf.divide(
            video,
            tf.constant(
                127.5,
                dtype=tf.float32)),
        tf.constant(
            1.,
            dtype=tf.float32))

    if use_bfloat16:
        ##### Conversion to bfloat16
        video = tf.image.convert_image_dtype(
            video,
            dtype=tf.bfloat16)

    return video
