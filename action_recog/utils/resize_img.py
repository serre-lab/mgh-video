import cv2

# Border pixel value
BLACK = [0, 0, 0]

def get_img_shape(img):
    '''Get image shape
    Args:
        img: 'Numpy' matrix
    Returns:
        'Integer' height and width of image
    '''
    return img.shape[0], img.shape[1]

def pad_images(img=None, is_height_big=True,
                TARGET_HEIGHT=224, TARGET_WIDTH=224):
    '''Pad image with zero pixels on both sides of the
        shorter dimension (between height and width)
    Args:
        img: 'Numpy' 3D array
        is_height_big: 'Bool' to specify if height
            is greater than width
        TARGET_HEIGHT: required height
        TARGET_WIDTH: required width
    Returns:
        Padded image of type 'Numpy'
    '''
    height, width = get_img_shape(img)

    if is_height_big:
        get_diff = TARGET_WIDTH - width
    
    else:
        get_diff = TARGET_HEIGHT - height

    pad_1 = get_diff / 2

    if get_diff % 2 == 0:
        pad_2 = pad_1

    else:
        pad_2 = pad_1 + 1

    if is_height_big:
        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            pad_1,
            pad_2,
            cv2.BORDER_CONSTANT,
            value=BLACK)

    else:
        img = cv2.copyMakeBorder(
            img,
            pad_1,
            pad_2,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=BLACK)

    return img

def resize(img=None, TARGET_HEIGHT=224, TARGET_WIDTH=224):
    '''Resize image to a given height
    and width while keeping constant
    aspect ratio.
    Args:
        img: 'Numpy' image
        TARGET_HEIGHT: required height
        TARGET_WIDTH: required width
    Returns:
        Image of type 'Numpy' resized to
        target height and width
    '''
    assert not img is None, '\'img\' is None'
    assert len(img.shape) <= 3, '\'img\' should be\
        2D or 3D. Found {}'.format(len(img.shape))
    assert type(TARGET_HEIGHT) == int, '\'TARGET_HEIGHT\'\
        should be an integer. Found to be {}'.format(type(TARGET_HEIGHT))
    assert type(TARGET_WIDTH) == int, '\'TARGET_WIDTH\'\
        should be an integer. Found to be {}'.format(type(TARGET_WIDTH))

    height, width = get_img_shape(img)
    # Resizing along longer dimension
    if height >= width:
        a_r = TARGET_HEIGHT / float(height)
        new_width = int(width * a_r)
        img = cv2.resize(
            img, (
                new_width,
                TARGET_HEIGHT))

    else:
        a_r = TARGET_WIDTH / float(width)
        new_height = int(height * a_r)
        img = cv2.resize(
            img, (
                TARGET_WIDTH,
                new_height))

    new_height, new_width = get_img_shape(img)
    # Padding images to make them square
    if new_height == TARGET_HEIGHT and new_width < TARGET_WIDTH:
        img = pad_images(
            img,
            is_height_big=True,
            TARGET_HEIGHT=TARGET_HEIGHT,
            TARGET_WIDTH=TARGET_WIDTH)

    elif new_height < TARGET_HEIGHT and new_width == TARGET_WIDTH:
        img = pad_images(
            img,
            is_height_big=False,
            TARGET_HEIGHT=TARGET_HEIGHT,
            TARGET_WIDTH=TARGET_WIDTH)
             
    return img

