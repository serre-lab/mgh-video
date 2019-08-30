from functools import partial

import tensorflow as tf

#TODO: implement true random brightness

def is_valid_crop_box(image,
                    mask=None,
                    height_diff=None,
                    width_diff=None,
                    target_image_size=[224, 224]):
    '''Get cropped image
    Args:
        image: '3D Tensor' [height, width, n_channels]
        mask: 'None' or '3D Tensor' [height, width, n_channels]
        height_diff: 'Tensor' to mention the height difference
                between target size and image height
        widht_diff: 'Tensor' to mention the width difference
                between target size and image width
        target_image_size: 'List' to mention target height
                and width
    Returns:
        '3D Tensor' of image and/or mask of same dtype as
        :image and :mask
    '''
    # Randomly get offset height and width within range
    # [0, height_diff], and [0, width, diff]
    offset_height = tf.random_uniform(
        shape=[],
        minval=tf.constant(0),
        maxval=height_diff,
        dtype=tf.int32)

    offset_width = tf.random_uniform(
        shape=[],
        minval=tf.constant(0),
        maxval=width_diff,
        dtype=tf.int32)

    # Get same crop for :image and :mask
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        target_image_size[0],
        target_image_size[1])

    if not mask is None:
        mask = tf.image.crop_to_bounding_box(
            mask,
            offset_height,
            offset_width,
            target_image_size[0],
            target_image_size[1])
    
    return image, mask

def is_not_valid_crop_box(image,
                        mask=None,
                        target_image_size=[224, 224]):
    '''Resize and pad image
    Args:
        image: '3D Tensor' [height, width, n_channels]
        mask: 'None' or '3D Tensor'
                [height, width, n_channels]
        target_image_size: 'List' to mention target
                height and width
    Returns:
        '3D Tensor' of image and/or mask of same dtype as
        :image and :mask
    '''
    image = tf.image.resize_image_with_crop_or_pad(
        image,
        target_image_size[0],
        target_image_size[1])

    if not mask is None:
        mask = tf.image.resize_image_with_crop_or_pad(
            mask,
            target_image_size[0],
            target_image_size[1])
        
        return image, mask
    else:
        return image, mask

def random_crop(image, height=None,
                width=None, mask=None,
                target_image_size=[224, 224]):
    '''Get random crop of image
    Args:
        image: '3D Tensor' [height, width, n_channels]
        height: 'Tensor' to specify the true height of
            :image
        width: 'Tensor' to specify the true width of
            :image
        mask: 'None' or '3D Tensor'
                [height, width, n_channels]
        target_image_size: 'List' of desired height and
            width
    Returns:
        Cropped image and/or mask of same dtype as :image
        and :mask
    '''
    height_diff = height - target_image_size[0]
    width_diff = width - target_image_size[1]

    # If image and mask dimensions are smaller than
    # target height or width, pad the image to bring
    # the dimensions up to the desired shape.
    # Otherwise, crop the image.
    image, mask = tf.cond(
        tf.math.logical_or(
            height_diff <= 0,
            width_diff <= 0), 
        lambda : is_not_valid_crop_box(
            image,
            mask=mask,
            target_image_size=target_image_size),
        lambda : is_valid_crop_box(
            image,
            mask=mask,
            height_diff=height_diff,
            width_diff=width_diff,
            target_image_size=target_image_size))

    return image, mask

def random_lr_flip(image, mask=None):
    '''Randomly flip image and/or mask
    Args:
        image: '3D Tensor' [height, width, n_channels]
        mask: 'None' or '3D Tensor' of same shape as :image
    Returns:
        Flipped image and/or mask if same dtype as :image
    '''
   
    get_image_shape = image.get_shape().as_list()

    # Merge :image and :mask along channel axis to ensure
    # random flipping is consistent between :image and
    # :mask.
    # merged tensor: [height, width, 2 * n_channels]
    if mask is None:
        merged = image
    else:
        merged = tf.concat(
            [image, mask], axis=-1)

    def __flip(merged):
        merged = tf.image.random_flip_left_right(
            merged)
        return merged

    def __noflip(merged):
        return merged

    # make flip function callable
    fl_fn = partial(__flip, merged)
    # Conditionally flip :merged left or right
    merged = tf.cond(
        tf.random_uniform(
            shape=[],
            dtype=tf.float32) > 0.5,
        lambda: __flip(merged),
        lambda: __noflip(merged))

    if mask is None:
        return merged, mask
    # Get back :image and :mask
    image = merged[:, :, :get_image_shape[-1]]
    mask = merged[:, :, get_image_shape[-1]:]
    
    return image, mask

# TODO: change adjust_brightness to random_brightness
def random_brightness(image):
    '''Apply brightness
    Args:
        image: '3D Tensor' [height, width, n_channels]
    Returns:
        '3D Tensor' of same dtype and shape
    '''

    # Range: [-0.15, 0.15)
    delta_var = tf.random_uniform(
        shape=[],
        dtype=tf.float32) * 0.2

    image = tf.image.adjust_brightness(
        image,
        delta=delta_var)

    return image

def random_contrast(image):
    '''Apply contrast
    Args:
        image: '3D Tensor' [height, width, n_channels]
    Returns:
        '3D Tensor' of same dtype and shape
    '''
    image = tf.image.random_contrast(
        image,
        0.9,
        1.1)

    return image

def random_gaussian_noise(image):
    '''Apply gaussina noise
    Args:
        image: '3D Tensor' [height, width, n_channels]
    Returns:
        '3D Tensor' of same dtype and shape
    '''

    def __add_noise(image):
        noise = tf.random_normal(
            shape=image.get_shape().as_list(),
            mean=0.0,
            stddev=0.03,
            dtype=tf.float32)
        image = tf.cast(
            image,
            tf.float32) / 255.0 + noise

        image = tf.cast(
            (
                tf.clip_by_value(
                    image,
                    0.0,
                    1.0)
                * 255.0),
            tf.uint8)

        return image
    
    def __no_noise(image):
        return image

    # make the gaussian noise adding process callable
    gn_fn = partial(__add_noise, image)
    # conditionally add noise
    image = tf.cond(
        tf.random_uniform(
            shape=[],
            dtype=tf.float32) > 0.5,
        lambda: __add_noise(image),
        lambda: __no_noise(image))
    
    return image
