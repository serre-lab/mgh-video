import tensorflow as tf

import augment_image_util as aiu

def preprocess_for_train(image=None,
                        height=None,
                        width=None,
                        target_image_size=[224, 224],
                        mask=None,
                        list_of_augmentations=[]):

    '''Preprocessing images during training'
    Args:
        image: '3D Tensor' [height, width, channels]
        height: 'Tensor' specifying true height of :image
        width: 'Tensor' specifying true width of :image
        target_image_size: 'List' to speify height and
            width of input image required by the model
        mask: Either 'None' or '3D Tensor'.
            For segmentation tasks, contains the mask
            which is a '3D Tensor' with shape equal to
            :images.
    Returns:
        Augmented '3D Tensor' image and/or mask of
            same dtype as :image. Returned mask will be None
            if :mask is None
    '''

    ##### Breaking conditions
    get_image_shape = image.get_shape().as_list()
    assert len(get_image_shape) == 3, 'Input shape length\
        should be 3. Found %d' %len(get_image_shape)

    if not mask is None:
        get_mask_shape = mask.get_shape().as_list()
        assert len(get_mask_shape) == 3, 'Shape of mask\
            should be 3. Found %d' %len(get_mask_shape)

    if len(list_of_augmentations) == 0:
        print('No augmentations mentioned, function will\
            return image and/or mask unchanged')

    ##### Call augmentation functions
    
    # Random cropping should be done for both image
    # and mask. Sending true height and width of :image
    # to set proper shapes. Following augmentations can
    # then be safely applied.
    if 'random_crop' in list_of_augmentations:
        image, mask = aiu.random_crop(
            image,
            height=height,
            width=width,
            mask=mask,
            target_image_size=target_image_size)

    # Random left-right flip should be done consistently among
    # :image and :mask
    if 'random_lr_flip' in list_of_augmentations:
        image, mask = aiu.random_lr_flip(
            image,
            mask=mask)

    # Random brightness, contrast and blur augmentations
    # are done on the image and NOT on the mask

    if 'random_brightness' in list_of_augmentations:
        image = aiu.random_brightness(
            image)
    
    if 'random_contrast' in list_of_augmentations:
        image = aiu.random_contrast(
            image)

    if 'random_gaussian_noise' in list_of_augmentations:
        image = aiu.random_gaussian_noise(
            image)

    # normalizing the image
    ## TODO: Add this as a flag into the training script
    #image = tf.cast(image,dtype=tf.float32) / 255. - 127.5
    return image, mask

def preprocess_for_eval(image=None,
                        height=None,
                        width=None,
                        target_image_size=[224, 224],
                        mask=None,
                        list_of_augmentations=[]):

    '''Preprocessing images during evaluation'
    Args:
        image: '3D Tensor' [height, width, channels]
        height: 'Tensor' specifying true height of :image
        width: 'Tensor' specifying true width of :image
        target_image_size: 'List' to specify height 
            and width of input image required by the model
        mask: Either 'None' or '3D Tensor'.
            For segmentation tasks, contains the mask
            which is a '3D Tensor' with shape equal to
            :images.
    Returns:
        Actual/Augmented '3D Tensor' image and/or mask of 
            same dtype as :image. Returned mask will be None
            if :mask is None
    '''

    ##### Breaking conditions   
    get_image_shape = image.get_shape().as_list()
    assert len(get_image_shape) == 3, 'Input shape length\
        should be 3. Found %d' %len(get_image_shape)

    if not mask is None:
        get_mask_shape = mask.get_shape().as_list()
        assert len(get_mask_shape) == 3, 'Shape of mask\
            should be 3. Found %d' %len(get_mask_shape)

    if len(list_of_augmentations) == 0:
        print('No augmentations mentioned, function will\
            return image and/or mask unchanged')

    ##### Call augmentation functions
    
    # Random cropping should be done for both image
    # and mask
    if 'random_crop' in list_of_augmentations:
        image, mask = aiu.random_crop(
            image,
            height=height,
            width=width,
            mask=mask,
            target_image_size=target_image_size)

    return image, mask


def preprocess_image(image=None,
                    height=None,
                    width=None,
                    mask=None,
                    is_training=False,
                    target_image_size=224,
                    use_bfloat16=False,
                    list_of_augmentations=[]):

    '''Preprocess the given image.
    Args:
        image: 'Tensor' representing an uint image of\
            arbitrary size
        height: 'Tensor to specify original height of\
            the image
        width: 'Tensor' to specify original width of\
            the image
        mask: Optional 'Tensor' to specify the mask\
            if model is trained for segmentation tasks.
            This function assumes that :image and\
            :mask have the same shape
        is_trainign: 'Boolean' to specify phase of\
            model session
        target_image_size: 'List' to represent\
            final input size accpeted by model. If\
            type 'Integer', then the value will be\
            used for both height and width of augmentd\
            image.
        use_bfloat16: 'Boolean' to specufy whether\
            to use dtype bfloat16. Saves memory and\
            improves computation efficiency
        list_of_augmentations: 'List' of augmentations\
            to be performed on the image
    Returns:
        Original/Augmented '3D Tensor' image and/or mask
    '''

    ##### Breaking conditions
    assert not image is None, 'image found to be None'
    assert not height is None, 'height found to be None'
    assert not width is None, 'width found to be None'

    if type(target_image_size) == int:
        target_image_size = [
            target_image_size,
            target_image_size]

    # Get back actual image
    image = tf.reshape(
        image,
        [
            height,
            width,
            3])

    # converting BGR to RGB colorspace
    image = image[..., ::-1]

    if not mask is None:
        mask = tf.reshape(
            mask,
            [
                height,
                width,
                1
            ])
        # this is just to control a labeling artefact of COCO stuff dataset
        # background is 0, and other labels fall in the range 92 - 182
        # must test this!!
        #mask = tf.nn.relu(tf.subtract(mask,91))

    if is_training:
        func = preprocess_for_train
    else:
        func = preprocess_for_eval

    image, mask = func(
        image=image,
        height=height,
        width=width,
        target_image_size=target_image_size,
        mask=mask,
        list_of_augmentations=list_of_augmentations)

    # Cast image/mask to float32
    image = tf.cast(
        image,
        tf.float32)
   
    if not mask is None:
        mask = tf.cast(
            mask,
            tf.int32)

    # Normalize image and/or mask

    # Conversion to bfloat16
    if use_bfloat16:
        image = tf.image.convert_image_dtype(
            image,
            dtype=tf.bfloat16)
        
        if not mask is None:
            mask = tf.image.convet_image_dtype(
                mask,
                dtype=tf.bfloat16)

    if mask is None:
        return image
    
    return image, mask
