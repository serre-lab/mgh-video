'''
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
arxiv: https://arxiv.org/abs/1705.07750
'''


import tensorflow as tf

from central_reservoir.utils.layers import linear
from central_reservoir.utils.layers import conv_batchnorm_relu
from central_reservoir.utils.layers import maxpool
from central_reservoir.utils.layers import avgpool

VALID_ENDPOINTS = (
    'Conv3d_1_7x7',
    'MaxPool3d_2a_3x3',
    'Conv3d_2b_1x1',
    'Conv3d_2c_3x3',
    'MaxPool3d_3a_3x3',
    'Mixed_3b',
    'Mixed_3c',
    'MaxPool3d_4a_3x3',
    'Mixed_4b',
    'Mixed_4c',
    'Mixed_4d',
    'Mixed_4e',
    'Mixed_4f',
    'MaxPool3d_5a_2x2',
    'Mixed_5b',
    'Mixed_5c',
    'Logits',
    'Predictions',
)

def build_i3d(only_imagenet=False, final_endpoint='Logits', use_batch_norm=False,
            use_cross_replica_batch_norm=False, num_classes=101,
            spatial_squeeze=True, dropout_keep_prob=1.0, num_cores=8):

    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

    def model(inputs, is_training):

        net = inputs
        end_points = {}

        print('Inputs: {}'.format(net.get_shape().as_list()))

        if only_imagenet:
            # 7x7x7 Conv, stride 2
            end_point = 'Conv3d_1a_7x7'
            net = conv_batchnorm_relu(net, end_point, 64,
                kernel_size=7, stride=2, is_training=is_training, num_cores=num_cores,
                use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # 1x3x3 Max-pool, stride 1, 2, 2
            end_point = 'MaxPool3d_2a_3x3'
            net = maxpool(net, end_point, ksize=[1, 1, 3, 3, 1],
                strides=[1, 1, 2, 2, 1], padding='SAME')
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # 1x1x1 Conv, stride 1
            end_point = 'Conv3d_2b_1x1'
            net = conv_batchnorm_relu(net, end_point, 64,
                kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # 3x3x3 Conv, stride 1
            end_point = 'Conv3d_2c_3x3'
            net = conv_batchnorm_relu(net, end_point, 192,
                kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                 use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # 1x3x3 Max-pool, stride 1, 2, 2
            end_point = 'MaxPool3d_3a_3x3'
            net = maxpool(net, end_point, ksize=[1, 1, 3, 3, 1],
                strides=[1, 1, 2, 2, 1], padding='SAME')
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 3b : Inception block
            end_point = 'Mixed_3b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 64,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 96,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 128,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 16,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 32,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 32,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 3c: Inception block
            end_point = 'Mixed_3c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 192,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 32,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 96,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-Pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stide 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 64,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # 3x3x3 Max-pool, stride 2, 2, 2
            end_point = 'MaxPool3d_4a_3x3'
            net = maxpool(net, end_point, ksize=[1, 3, 3, 3, 1],
                strides=[1, 2, 2, 2, 1], padding='SAME')
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 4b: Inception block
            end_point = 'Mixed_4b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 192,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 96,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 208,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 16,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 48,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 64,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 4c: Inception block
            end_point = 'Mixed_4c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 160,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 112,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 224,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 24,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 64,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 64,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 4d: Inception block
            end_point = 'Mixed_4d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 256,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 24,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 64,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 64,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 4e: Inception block
            end_point = 'Mixed_4e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 112,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 144,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 288,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 32,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 64,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 64,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 4f: Inception block
            end_point = 'Mixed_4f'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 256,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 160,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 320,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 32,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 128,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # 2x2x2 Max-pool, stride 2x2x2
            end_point = 'MaxPool3d_5a_2x2'
            net = maxpool(net, end_point, ksize=[1, 2, 2, 2, 1],
                strides=[1, 2, 2, 2, 1], padding='SAME')
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 5b: Inception block
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 256,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 160,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 320,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 32,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0a_3x3', 128,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points

            # Mixed 5c: Inception block
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    branch_0 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 384,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 192,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_1 = conv_batchnorm_relu(branch_1, 'Conv3d_0b_3x3', 384,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(net, 'Conv3d_0a_1x1', 48,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)
                    # 3x3x3 Conv, stride 1
                    branch_2 = conv_batchnorm_relu(branch_2, 'Conv3d_0b_3x3', 128,
                        kernel_size=3, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    branch_3 = maxpool(net, 'MaxPool3d_0a_3x3',
                        ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
                        padding='SAME')
                    # 1x1x1 Conv, stride 1
                    branch_3 = conv_batchnorm_relu(branch_3, 'Conv3d_0b_1x1', 128,
                        kernel_size=1, stride=1, is_training=is_training, num_cores=num_cores,
                         use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm)

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
            get_shape = net.get_shape().as_list()
            print('{} : {}'.format(end_point, get_shape))

            end_points[end_point] = net
            if final_endpoint == end_point: return net, end_points
            # net: [N, T/8, 1, 1, 1024]
            # Logits
            end_point = 'Logits'
            with tf.variable_scope(end_point):
                # 2x7x7 Average-pool, stride 1, 1, 1
                net = avgpool(net, ksize=[1, 2, 7, 7, 1],
                    strides=[1, 1, 1, 1, 1], padding='VALID')
                get_shape = net.get_shape().as_list()
                print('{} / Average-pool3D: {}'.format(end_point, get_shape))
                end_points[end_point + '_average_pool3d'] = net

                # net: [N, T/16, 1, 1, 1024]
                # Dropout
                net = tf.nn.dropout(net, dropout_keep_prob)

                # 1x1x1 Conv, stride 1
                logits = conv_batchnorm_relu(net, 'Conv3d_0c_1x1', num_classes,
                    kernel_size=1, stride=1, activation=None,
                    use_batch_norm=False, use_cross_replica_batch_norm=False,
                    is_training=is_training, num_cores=num_cores)
                get_shape = logits.get_shape().as_list()
                print('{} / Conv3d_0c_1x1 : {}'.format(end_point, get_shape))
                # logits: [N, T/16, 1, 1, n_classes]

                if spatial_squeeze:
                    # Removes dimensions of size 1 from the shape of a tensor
                    # Specify which dimensions have to be removed: 2 and 3
                    logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
                    get_shape = logits.get_shape().as_list()
                    print('{} / Spatial Squeeze : {}'.format(end_point, get_shape))
                    # [N, T/16, n_classes]

            averaged_logits = tf.reduce_mean(logits, axis=1) # [N, num_classes]
            get_shape = averaged_logits.get_shape().as_list()
            print('{} / Averaged Logits : {}'.format(end_point, get_shape))

            end_points[end_point] = averaged_logits
            if final_endpoint == end_point: return averaged_logits, end_points

            # Predictions
            end_point = 'Predictions'
            predictions = tf.nn.softmax(
                averaged_logits)
            end_points[end_point] = predictions
            return predictions, end_points

    return model

def InceptionI3d(only_imagenet=True, final_endpoint='Logits', use_batch_norm=False,
                use_cross_replica_batch_norm=False, num_classes=101,
                spatial_squeeze=True, num_cores=8,
                dropout_keep_prob=1.0):

    return build_i3d(
        only_imagenet=only_imagenet,
        final_endpoint=final_endpoint,
        use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm,
        num_cores=num_cores,
        num_classes=num_classes,
        spatial_squeeze=spatial_squeeze,
        dropout_keep_prob=dropout_keep_prob)


