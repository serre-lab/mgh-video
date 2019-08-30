'''
ResUnet: UNET + skip connections
UNET paper: arxiv: https://arxiv.org/abs/1505.04597
'''

import tensorflow as tf

from central_reservoir.utils.layers import linear
from central_reservoir.utils.layers import conv_batchnorm_relu
from central_reservoir.utils.layers import upconv_2D
from central_reservoir.utils.layers import maxpool
from central_reservoir.utils.layers import avgpool

VALID_ENDPOINTS = (
    'Encode_1',
    'Encode_2',
    'Encode_3',
    'Encode_4',
    'Encode_5',
    'Decode_1',
    'Decode_2',
    'Decode_3',
    'Decode_4',
)

layer_outputs = {}

def resnet_block(model_scope_name='resunet_1',
                end_point='',
                scope_id=0,
                inputs=None,
                out_channels=64,
                is_training=True,
                num_cores=8,
                use_batch_norm=False,
                use_cross_replica_batch_norm=False):

    # Shortcut: 3x3 Conv, padding='same', no activation
    shortcut_conv2d = conv_batchnorm_relu(inputs,
        'shortcut_Conv2d_{}a'.format(scope_id),
        out_channels, kernel_size=3, stride=1, padding='SAME',
        activation=None, is_training=is_training,
        num_cores=num_cores, use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm)
    get_shape = shortcut_conv2d.get_shape().as_list()
    full_layer_name = model_scope_name + '/' + end_point + '/' +\
            'shortcut_Conv2d_{}a'.format(scope_id)
    print('{}: {}'.format(full_layer_name, get_shape))
    layer_outputs[full_layer_name] = shortcut_conv2d

    # 3x3 Conv, padding='same'
    conv2d_1 = conv_batchnorm_relu(inputs, 'Conv2d_{}a'.format(scope_id),
        out_channels, kernel_size=3, stride=1, padding='SAME',
        is_training=is_training, num_cores=num_cores,
        use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm)
    get_shape = conv2d_1.get_shape().as_list()
    full_layer_name = model_scope_name + '/' + end_point + '/'\
            + 'Conv2d_{}a'.format(scope_id)
    print('{}: {}'.format(full_layer_name, get_shape))
    layer_outputs[full_layer_name] = conv2d_1

    # 3x3 Conv, padding='same', no activation
    conv2d_2 = conv_batchnorm_relu(conv2d_1, 'Conv2d_{}b'.format(scope_id),
        out_channels, kernel_size=3, stride=1, padding='SAME',
        is_training=is_training, num_cores=num_cores, activation=None,
        use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm)
    get_shape = conv2d_2.get_shape().as_list()
    full_layer_name = model_scope_name + '/' + end_point + '/' +\
            'Conv2d_{}b'.format(scope_id)
    print('{}: {}'.format(full_layer_name, get_shape))
    layer_outputs[full_layer_name] = conv2d_2

    # Sum: x + f(x) and relu
    with tf.variable_scope('sum_relu'):
        sum_1 = tf.nn.relu(
            shortcut_conv2d + conv2d_2)
    get_shape = sum_1.get_shape().as_list()
    full_layer_name = model_scope_name + '/' + end_point + '/' + 'sum_relu'
    print('{}: {}'.format(full_layer_name, get_shape))
    layer_outputs[full_layer_name] = sum_1

    return sum_1

def build_resUnet(final_endpoint='Decode_4',
                use_batch_norm=False,
                use_cross_replica_batch_norm=False,
                num_cores=8,
                num_classes=92,
                model_scope_name='resunet'):

    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError('Unknown final endpoint %s' %final_endpoint)

    def model(inputs, is_training):

        net = inputs

        print('Input: {}'.format(net.get_shape().as_list()))

        with tf.variable_scope(model_scope_name):
            # Encode_1
            end_point = 'Encode_1'
            with tf.variable_scope(end_point):
                # 3x3 Conv, padding='same'
                conv2d_1a = conv_batchnorm_relu(net, 'Conv2d_1a', num_classes,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_1a

                # 3x3 Conv, padding='same'
                conv2d_1b = conv_batchnorm_relu(conv2d_1a, 'Conv2d_1b', num_classes,
                    kernel_size=3, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                get_shape = conv2d_1b.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_1b'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_1b

                # 2x2 MaxPool
                maxpool_1a = maxpool(conv2d_1b, 'MaxPool_1a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_1a
           
            if final_endpoint == end_point: return maxpool_1a, layer_outputs


            # Encode_2
            end_point = 'Encode_2'
            with tf.variable_scope(end_point):
                sum_2a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=2,
                    inputs=maxpool_1a,
                    out_channels=num_classes*2,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                
                maxpool_2a = maxpool(sum_2a, 'MaxPool_2a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_2a

            if final_endpoint == end_point: return maxpool_2a, layer_outputs


            # Encode_3
            end_point = 'Encode_3'
            with tf.variable_scope(end_point):
                sum_3a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=3,
                    inputs=maxpool_2a,
                    out_channels=num_classes*4,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                
                maxpool_3a = maxpool(sum_3a, 'MaxPool_3a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_3a

            if final_endpoint == end_point: return maxpool_3a, layer_outputs


            # Encode_4
            end_point = 'Encode_4'
            with tf.variable_scope(end_point):
                sum_4a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=4,
                    inputs=maxpool_3a,
                    out_channels=num_classes*8,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                
                maxpool_4a = maxpool(sum_4a, 'MaxPool_4a',
                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding='SAME')
                get_shape = maxpool_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'MaxPool_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = maxpool_4a

            if final_endpoint == end_point: return maxpool_3a, layer_outputs


            # Encode_5
            end_point = 'Encode_5'
            with tf.variable_scope(end_point):
                sum_5a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=4,
                    inputs=maxpool_4a,
                    out_channels=num_classes*16,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)

            if final_endpoint == end_point: return conv2d_5b, layer_outputs


            # Decode_1
            end_point = 'Decode_1'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_1a = upconv_2D(sum_5a, 'UpConv2d_1a', num_classes*8,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_1a

                # Merge
                merge_1a = tf.concat(
                    [sum_4a, upconv2d_1a],
                    axis=-1,
                    name='merge_1a')
                get_shape = merge_1a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_1a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_1a

                # Residual block
                sum_d_1a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=1,
                    inputs=merge_1a,
                    out_channels=num_classes*8,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)

            if final_endpoint == end_point: return sum_d_1a, layer_outputs

     
            # Decode_2
            end_point = 'Decode_2'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_2a = upconv_2D(sum_d_1a, 'UpConv2d_2a', num_classes*4,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_2a

                # Merge
                merge_2a = tf.concat(
                    [sum_3a, upconv2d_2a],
                    axis=-1,
                    name='merge_2a')
                get_shape = merge_2a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_2a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_2a

                # Residual block
                sum_d_2a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=2,
                    inputs=merge_2a,
                    out_channels=num_classes*4,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)

            if final_endpoint == end_point: return sum_d_2a, layer_outputs


            # Decode_3
            end_point = 'Decode_3'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_3a = upconv_2D(sum_d_2a, 'UpConv2d_3a', num_classes*2,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_3a

                # Merge
                merge_3a = tf.concat(
                    [sum_2a, upconv2d_3a],
                    axis=-1,
                    name='merge_3a')
                get_shape = merge_3a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_3a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_3a

                # Residual block
                sum_d_3a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=3,
                    inputs=merge_3a,
                    out_channels=num_classes*2,
                    is_training=is_training,
                    num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)

            if final_endpoint == end_point: return sum_d_3a, layer_outputs

            
            # Decode_4
            end_point = 'Decode_4'
            with tf.variable_scope(end_point):
                # Up-convolution
                upconv2d_4a = upconv_2D(sum_d_3a, 'UpConv2d_4a', num_classes,
                    kernel_size=(2, 2), strides=(2, 2), use_bias=True,
                    padding='valid')
                get_shape = upconv2d_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'UpConv2d_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = upconv2d_4a

                # Merge
                merge_4a = tf.concat(
                    [conv2d_1b, upconv2d_4a],
                    axis=-1,
                    name='merge_4a')
                get_shape = merge_4a.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'merge_4a'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = merge_4a

                # Residual block
                sum_d_4a = resnet_block(
                    model_scope_name,
                    end_point,
                    scope_id=4,
                    inputs=merge_4a,
                    out_channels=num_classes,
                    is_training=is_training,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # 1x1 Conv, padding='same'
                conv2d_d_4c = conv_batchnorm_relu(sum_d_4a, 'Conv2d_d_4c', 93,
                    kernel_size=1, stride=1, padding='SAME',
                    is_training=is_training, num_cores=num_cores,
                    use_batch_norm=use_batch_norm,
                    use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                    activation=None)
                get_shape = conv2d_d_4c.get_shape().as_list()
                full_layer_name = model_scope_name + '/' + end_point + '/' + 'Conv2d_d_4c'
                print('{}: {}'.format(full_layer_name, get_shape))
                layer_outputs[full_layer_name] = conv2d_d_4c

            layer_outputs[end_point] = conv2d_d_4c
            if final_endpoint == end_point: return conv2d_d_4c, layer_outputs

    return model


def RESUNET(final_endpoint='Decode_4', use_batch_norm=False,
        use_cross_replica_batch_norm=False, num_cores=8, num_classes=92,
        model_scope_name='resunet_1'):

    return build_resUnet(
        final_endpoint=final_endpoint,
        use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm,
        num_cores=num_cores,
        num_classes=num_classes,
        model_scope_name=model_scope_name)
