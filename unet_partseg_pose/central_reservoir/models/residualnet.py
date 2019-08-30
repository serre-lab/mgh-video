'''
Convolutional Networks for Biomedical Image Segmentation
arxiv: https://arxiv.org/abs/1505.04597
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
    'Pyr_5',
    'Pyr_4',
    'Pyr_3',
    'Pyr_2',
    'Head_1',
    'Head_2',
    'Head_3',
    'Head_4',
    'Head_Master'
)

def nearest_upsampling(data, scale):
    """Nearest neighbor upsampling implementation.
    Args:
        data: A tensor with a shape of [batch, height_in, width_in, channels].
        scale: An integer multiple to scale resolution of input data.
    Returns:
        data_up: A tensor with a shape of
        [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
        data.
    """
    with tf.name_scope('nearest_upsampling'):
        bs, h, w, c = data.get_shape().as_list()
        bs = -1 if bs is None else bs
        # Use reshape to quickly upsample the input.  The nearest pixel is selected
        # implicitly via broadcasting.
        data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
            [1, 1, scale, 1, scale, 1], dtype=data.dtype)
        return tf.reshape(data, [bs, h * scale, w * scale, c])


def residual_path(input_var, out_channels,
            scope=None,
            stride=[1,1],
            kernel_sizes=[1,3],
            activation=tf.keras.activations.relu,
            padding='SAME',
            spectral_norm_flag=False,
            update_collection=False,
            is_training=True,
            use_batch_norm=False,
            use_cross_replica_batch_norm=False,
            num_cores=8,
            initializer=tf.keras.initializers.glorot_normal):
    
    with tf.variable_scope(scope):
        # first conv
        output_var = conv_batchnorm_relu(input_var, 'Conv2d_Unit1', out_channels,
                            kernel_size=kernel_sizes[0],
                            stride=stride[0],
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding)
        # second conv
        output_var =  conv_batchnorm_relu(output_var, 'Conv2d_Unit2', out_channels,
                            kernel_size=kernel_sizes[1],
                            stride=stride[1],
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding)
        # after this technically we need to merge other paths with a 1x1 convolution
        return output_var

# split into 'cardinality' number of channels
# then transform each channel independently
# merge them via a 1x1 convolution while downsampling spatial resolution
# add in a skip from the input before rectification
def resnext_layer(input_var, out_channels, cardinality,
            scope=None,
            stride=[1,2,1],
            kernel_sizes=[1,3,1],
            activation=tf.keras.activations.relu,
            padding='SAME',
            spectral_norm_flag=False,
            update_collection=False,
            is_training=True,
            use_batch_norm=False,
            use_cross_replica_batch_norm=False,
            num_cores=8,
            initializer=tf.keras.initializers.glorot_normal):
    with tf.variable_scope(scope):
        input_shape = input_var.get_shape().as_list()
        # get number of channels in input
        ch = input_shape[-1]
        # group conv out_channels
        group_conv_channels = out_channels // 2
        # create a list of split layers
        split_layers = list()
        for path in range(cardinality):
            current_split = residual_path(input_var, group_conv_channels // cardinality,
                                scope='Path'+str(path),
                                stride=stride[:2],
                                kernel_sizes=kernel_sizes[:2],
                                activation=activation,
                                padding=padding,
                                spectral_norm_flag=spectral_norm_flag,
                                update_collection=update_collection,
                                is_training=is_training,
                                use_batch_norm=use_batch_norm,
                                use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                                num_cores=num_cores,
                                initializer=initializer)
            # append to the list
            split_layers.append(current_split)


        # concatenate all the splits
        grouped_input = tf.concat(split_layers,axis=3)
        # 1x1 convolution. IMP: No relu here!
        merged_input = conv_batchnorm_relu(grouped_input, 'Merge', out_channels,
                            kernel_size=kernel_sizes[-1],
                            stride=stride[-1],
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding,
                            activation=None)
        sh = merged_input.get_shape().as_list()
        if not(sh[1] == input_shape[1]):
            mod_input = conv_batchnorm_relu(input_var, 'Downsample', out_channels,
                            kernel_size=1,
                            stride=2,
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding,
                            activation=None)
        else:
            mod_input = input_var

        # add the skip connection and the rectify!
        return tf.nn.relu(mod_input + merged_input,name='rectify')

def resnet_layer(input_var, out_channels, cardinality,
            scope=None,
            stride=[1,2,1],
            kernel_sizes=[1,3,1],
            activation=tf.keras.activations.relu,
            padding='SAME',
            spectral_norm_flag=False,
            update_collection=False,
            is_training=True,
            use_batch_norm=False,
            use_cross_replica_batch_norm=False,
            num_cores=8,
            initializer=tf.keras.initializers.glorot_normal):
    with tf.variable_scope(scope):
        input_shape = input_var.get_shape().as_list()
        # get number of channels in input
        ch = input_shape[-1]
        # group conv out_channels
        group_conv_channels = out_channels // 2
        output_var = conv_batchnorm_relu(input_var, 'Conv2d_Unit1', group_conv_channels,
                            kernel_size=kernel_sizes[0],
                            stride=stride[0],
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding)
        # second conv
        output_var =  conv_batchnorm_relu(output_var, 'Conv2d_Unit2', group_conv_channels,
                            kernel_size=kernel_sizes[1],
                            stride=stride[1],
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding)
 
        # third conv
        output_var =  conv_batchnorm_relu(output_var, 'Conv2d_Unit3', out_channels,
                            kernel_size=kernel_sizes[2],
                            stride=stride[2],
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding,
                            activation=None)
 
        sh = output_var.get_shape().as_list()
        if not(sh[1] == input_shape[1]):
            mod_input = conv_batchnorm_relu(input_var, 'Downsample', out_channels,
                            kernel_size=1,
                            stride=2,
                            is_training=is_training,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            num_cores=num_cores,
                            padding=padding,
                            activation=None)
        else:
            mod_input = input_var

        # add the skip connection and the rectify!
        return tf.nn.relu(mod_input + output_var,name='rectify')

def residual_stack(input_var, out_channels, cardinality,
            num_stacks=1,
            scope=None,
            should_stride=True,
            activation=tf.keras.activations.relu,
            padding='SAME',
            spectral_norm_flag=False,
            update_collection=False,
            is_training=True,
            use_batch_norm=False,
            use_cross_replica_batch_norm=False,
            num_cores=8,
            initializer=tf.keras.initializers.glorot_normal,
            model_type='resnet'):
    
    with tf.variable_scope(scope):
        layer = input_var
        for stack in range(num_stacks):
            if (stack == 0) and should_stride:
                flag = True
                stride = [1,2,1]
            else:
                flag = False
                stride = [1,1,1]
            ####
            # account for the different models here
            ####

            if 'resnet' in model_type:
                cur_output = resnet_layer(layer, out_channels, cardinality,
                        scope='Layer_'+str(stack),
                        stride=stride,
                        activation=activation,
                        padding=padding,
                        spectral_norm_flag=spectral_norm_flag,
                        update_collection=update_collection,
                        is_training=is_training,
                        use_batch_norm=use_batch_norm,
                        use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                        num_cores=num_cores,
                        initializer=initializer)
            elif 'resnext' in model_type:
                cur_output = resnext_layer(layer, out_channels, cardinality,
                        scope='Layer_'+str(stack),
                        stride=stride,
                        activation=activation,
                        padding=padding,
                        spectral_norm_flag=spectral_norm_flag,
                        update_collection=update_collection,
                        is_training=is_training,
                        use_batch_norm=use_batch_norm,
                        use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                        num_cores=num_cores,
                        initializer=initializer)
            else:
                raise ValueError('This model type is currently not supported. Use [resnet|resnet150|resnext|resnext150]')

            if flag:
                # need to do some dimension handling
                pool_inp = tf.layers.average_pooling2d(inputs=layer, pool_size=[2,2], strides=2, padding='SAME')
                pool_inp = conv_batchnorm_relu(pool_inp, 'skip1x1_layer'+str(stack), cur_output.get_shape().as_list()[-1],
                                        stride=1, kernel_size=1, is_training=is_training,
                                        use_batch_norm=use_batch_norm, use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                                        num_cores=num_cores, activation=None)
            else:
                # dimensions match!
                pool_inp = layer
            # add the skip connection
            layer = tf.nn.relu(pool_inp + cur_output)

        return layer

def residual_network(inputs,is_training,
                num_cores=8,
                use_batch_norm=False,
                use_cross_replica_batch_norm=False,
                final_endpoint='Head_Master',
                model_type='resnet',
                stack_architecture=[3,4,6,3]):

    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError('Unknown final endpoint %s' %final_endpoint)

    net = inputs
    end_points = {}

    print('Input: {}'.format(net.get_shape().as_list()))

    # Encode_1
    end_point = 'Encode_1'
    out_channels_1a = 64
    with tf.variable_scope(end_point):
        # 7x7 Conv, padding='same'
        encode_1 = conv_batchnorm_relu(net, 'Conv2d_1a', out_channels_1a,
            kernel_size=7, stride=2, padding='SAME',
            is_training=is_training, num_cores=num_cores,
            use_batch_norm=use_batch_norm,
            use_cross_replica_batch_norm=use_cross_replica_batch_norm)

        get_shape = encode_1.get_shape().as_list()
        print('{} / Encode_1: {}'.format(end_point, get_shape))
        
    end_points[end_point] = encode_1
    if final_endpoint == end_point: return encode_1, end_points

    if len(stack_architecture) < 4:
        raise ValueError('Not enough hyperparameter specifications. This module needs at least 4 blocks of stacks to be defined')

    # Encode_2
    end_point = 'Encode_2'
    cardinality=32
    # 3
    encode_2 = residual_stack(encode_1, 256, cardinality,
            num_stacks=stack_architecture[0], scope=end_point, should_stride=True,
            is_training=is_training,
            use_batch_norm=use_batch_norm,
            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
            num_cores=num_cores,
            model_type=model_type)
    get_shape = encode_2.get_shape().as_list()
    print('{} / Encode_2: {}'.format(end_point, get_shape))
    end_points[end_point] = encode_2
    if final_endpoint == end_point: return encode_2, end_points


    # Encode_3
    end_point = 'Encode_3'
    # 4
    encode_3 = residual_stack(encode_2, 512, cardinality,
            num_stacks=stack_architecture[1], scope=end_point, should_stride=True,
            is_training=is_training,
            use_batch_norm=use_batch_norm,
            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
            num_cores=num_cores,
            model_type=model_type)

    get_shape = encode_3.get_shape().as_list()
    print('{} / Encode_3: {}'.format(end_point, get_shape))
    end_points[end_point] = encode_3
    if final_endpoint == end_point: return encode_3, end_points


    # Encode_4
    end_point = 'Encode_4'
    # 6
    encode_4 = residual_stack(encode_3, 1024, cardinality,
            num_stacks=stack_architecture[2], scope=end_point, should_stride=True,
            is_training=is_training,
            use_batch_norm=use_batch_norm,
            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
            num_cores=num_cores,
            model_type=model_type)

    get_shape = encode_4.get_shape().as_list()
    print('{} / Encode_4: {}'.format(end_point, get_shape))
    end_points[end_point] = encode_4
    if final_endpoint == end_point: return encode_4, end_points


    # Encode_5
    end_point = 'Encode_5'
    # 3
    encode_5 = residual_stack(encode_4, 2048, cardinality,
            num_stacks=stack_architecture[3], scope=end_point, should_stride=True,
            is_training=is_training,
            use_batch_norm=use_batch_norm,
            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
            num_cores=num_cores,
            model_type=model_type)

    get_shape = encode_5.get_shape().as_list()
    print('{} / Encode_5: {}'.format(end_point, get_shape))
    end_points[end_point] = encode_5
    if final_endpoint == end_point: return encode_5, end_points

    return end_points
    

def network_head(input_var, out_channels,
            stride=[1,1],
            kernel_sizes=[3,3],
            activation=tf.keras.activations.relu,
            padding='SAME',
            spectral_norm_flag=False,
            update_collection=False,
            is_training=True,
            use_batch_norm=False,
            use_cross_replica_batch_norm=False,
            num_cores=8,
            initializer=tf.keras.initializers.glorot_normal):
    
    intermediate_layer = conv_batchnorm_relu(input_var, 'Conv2d_3x3_a', out_channels,
                             kernel_size=kernel_sizes[0], stride=stride[0], padding=padding,
                             is_training=is_training, num_cores=num_cores,
                             use_batch_norm=use_batch_norm,
                             use_cross_replica_batch_norm=use_cross_replica_batch_norm)
    head = conv_batchnorm_relu(intermediate_layer, 'Conv2d_3x3_b', out_channels,
                             kernel_size=kernel_sizes[0], stride=stride[0], padding=padding,
                             is_training=is_training, num_cores=num_cores,
                             use_batch_norm=use_batch_norm,
                             use_cross_replica_batch_norm=use_cross_replica_batch_norm)
    return head

def build_residualnetwork_fpn(final_endpoint='Head_Master',
                use_batch_norm=False,
                use_cross_replica_batch_norm=False,
                num_cores=8,
                num_classes=92,
                model_type='resnet'):

    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError('Unknown final endpoint %s' %final_endpoint)

    def model(inputs, is_training):

        net = inputs
        end_points = {}

        print('Input: {}'.format(net.get_shape().as_list()))
        
        '''
        Build the feedforward part of the network
        Here is where we select the model type.
        Currently supported options: resnet, resnet150, resnext, resnext150
        '''
        stack_architecture = [3, 4, 6, 3]
        # define the stacking hyperparameters for the larger model type
        if '150' in model_type:
            stack_architecture = [3, 8, 36, 3]

        _, layers = residual_network(inputs, is_training,
                            num_cores=num_cores,
                            use_batch_norm=use_batch_norm,
                            use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                            final_endpoint='Encode_5',
                            model_type=model_type,
                            stack_architecture=stack_architecture)

        #########
        # Feature Pyramid
        #########
        heads = {}
        end_point = 'Pyr_5'
        with tf.variable_scope(end_point):
            fpn_5 = conv_batchnorm_relu(layers['Encode_5'], 'Conv2d_1x1', 256,
                             kernel_size=1, stride=1, padding='SAME',
                             is_training=is_training, num_cores=num_cores,
                             use_batch_norm=use_batch_norm,
                             use_cross_replica_batch_norm=use_cross_replica_batch_norm)
            fpn_5_head = network_head(fpn_5, 128, is_training=is_training)
            heads[end_point] = fpn_5_head
        get_shape = fpn_5.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))

        end_point = 'Pyr_4'
        with tf.variable_scope(end_point):
            fpn_4 = conv_batchnorm_relu(layers['Encode_4'], 'Conv2d_1x1', 256,
                             kernel_size=1, stride=1, padding='SAME',
                             is_training=is_training, num_cores=num_cores,
                             use_batch_norm=use_batch_norm,
                             use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                             activation=None)
            # upsample fpn_5 and add
            sh = fpn_5.get_shape().as_list()
            #fpn_4 = tf.nn.relu(fpn_4 + tf.image.resize_images(fpn_5, [sh[1]*2, sh[2]*2]))
            fpn_4 = tf.nn.relu(fpn_4 + nearest_upsampling(fpn_5, 2))
            fpn_4_head = network_head(tf.nn.relu(fpn_4), 128, is_training=is_training)
            heads[end_point] = fpn_4_head
        get_shape = fpn_4.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))


        end_point = 'Pyr_3'
        with tf.variable_scope(end_point):
            fpn_3 = conv_batchnorm_relu(layers['Encode_3'], 'Conv2d_1x1', 256,
                             kernel_size=1, stride=1, padding='SAME',
                             is_training=is_training, num_cores=num_cores,
                             use_batch_norm=use_batch_norm,
                             use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                             activation=None)
            # upsample fpn_4 and add
            sh = fpn_4.get_shape().as_list()
            #fpn_3 = tf.nn.relu(fpn_3 + tf.image.resize_images(fpn_4, [sh[1]*2, sh[2]*2]))
            fpn_3 = tf.nn.relu(fpn_3 + nearest_upsampling(fpn_4, 2))
            fpn_3_head = network_head(tf.nn.relu(fpn_3), 128, is_training=is_training)
            heads[end_point] = fpn_3_head
        get_shape = fpn_3.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))


        end_point = 'Pyr_2'
        with tf.variable_scope(end_point):
            fpn_2 = conv_batchnorm_relu(layers['Encode_2'], 'Conv2d_1x1', 256,
                             kernel_size=1, stride=1, padding='SAME',
                             is_training=is_training, num_cores=num_cores,
                             use_batch_norm=use_batch_norm,
                             use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                             activation=None)
            # upsample fpn_3 and add
            sh = fpn_3.get_shape().as_list()
            #fpn_2 = tf.nn.relu(fpn_2 + tf.image.resize_images(fpn_3, [sh[1]*2, sh[2]*2]))
            fpn_2 = tf.nn.relu(fpn_2 + nearest_upsampling(fpn_3, 2))
            fpn_2_head = network_head(tf.nn.relu(fpn_2), 128, is_training=is_training)
            heads[end_point] = fpn_2_head
        get_shape = fpn_2.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))
 
        end_point = 'Head_Master'
        with tf.variable_scope(end_point):
            net_heads = list()
            inp_shape = inputs.get_shape().as_list()
            for head_name in heads.keys():
                sh = heads[head_name].get_shape().as_list()
                # calculate scale at which upsampling must be done
                scale_factor = inp_shape[1] // sh[1]
                #up_sampled_layer = tf.image.resize_images(heads[head_name],[inp_shape[1], inp_shape[2]])
                up_sampled_layer = nearest_upsampling(heads[head_name],scale_factor)
                net_heads.append(up_sampled_layer)

            # concatenate
            concat_head = tf.concat(net_heads,axis=3)
            model_output = conv_batchnorm_relu(concat_head, 'Conv2d_3x3_Final', num_classes,
                                    kernel_size=3, stride=1, padding='SAME',
                                    is_training=is_training, num_cores=num_cores,
                                    use_batch_norm=use_batch_norm,
                                    use_cross_replica_batch_norm=use_cross_replica_batch_norm,
                                    activation=None)
        get_shape = model_output.get_shape().as_list()
        print('{} : {}'.format(end_point, get_shape))


        return model_output

    return model


def ResidualNetwork_FPN(final_endpoint='Head_Master', use_batch_norm=False,
        use_cross_replica_batch_norm=False, num_cores=8, num_classes=92,
        model_type='resnet'):

    return build_residualnetwork_fpn(
        final_endpoint=final_endpoint,
        use_batch_norm=use_batch_norm,
        use_cross_replica_batch_norm=use_cross_replica_batch_norm,
        num_cores=num_cores,
        num_classes=num_classes,
        model_type=model_type)
