'''Implementation of cross shard
batch-normalization operation on TPUs.
Main author: Dingdong Yang
Collab: Rohit Saha
'''

import tensorflow as tf

class ExponentialMovingAverage():
    def __init__(self, name, ch, 
                input_feat, decay=0.9):
        value_initialized = 1.0 if name == 'moving_variance' else 0.0

        self.input_feat = input_feat
        get_input_feat_shape = self.input_feat.get_shape().as_list()

        shape = [1, 1, 1, ch]\
            if len(get_input_feat_shape) == 4\
            else [1, 1, 1, 1, ch]

        self.decay = 0.9
        self.moving_stat = tf.get_variable(
            name,
            initializer=lambda: tf.constant(
                value_initialized,
                shape=shape),
            trainable=False)

    def __call__(self, input_var):
        with tf.colocate_with(self.moving_stat):
            update_stat = self.moving_stat\
                            * self.decay\
                            + input_var\
                            * (1 - self.decay)

        assign_op = tf.assign(
            self.moving_stat,
            update_stat).op
        assign_op._unconditional_update = False

        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            assign_op)


def cross_replica_batch_norm(input_feat,
                        is_training=True,
                        scope='bn',
                        center=True,
                        scale=True,
                        epsilon=1e-3,
                        core_num=8):

    '''Cross replica batch normalization

    Args:
        input_feat: 4D or 5D Tensor
        is_training: Boolean to mention phase
        scope: String to mention name
        center: Boolean to specify centering
        scale: Boolean to specify scaling
        epsilon: Float var
        core_num: Integer to denote number of cores

    Return: Normalized Tensor of input_feat shape
    '''

    crp_sum = tf.contrib.tpu.cross_replica_sum
    with tf.variable_scope(
        scope,
        reuse=tf.AUTO_REUSE):

        shape = input_feat.get_shape().as_list()
        shape = [1, 1, 1, shape[-1]]\
            if len(shape) == 4\
            else [1, 1, 1, 1, shape[-1]]

        if center:
            beta = tf.Variable(
                lambda: tf.constant(
                    0.0,
                    shape=shape),
                name='beta',
                trainable=True)
        else:
            beta = tf.constant(
                0.0,
                shape=shape,
                name='beta')

        if scale:
            gamma = tf.Variable(
                lambda: tf.constant(
                    1.0,
                    shape=shape),
                name='gamma',
                trainable=True)
        else:
            gamma = tf.constant(
                1.0,
                shape=shape,
            name='gamma')

        # Moving average instance initialization
        ema_mean = ExponentialMovingAverage(
            'moving_mean',
            shape[-1],
            input_feat,
            decay=0.9)
        ema_var = ExponentialMovingAverage(
            'moving_variance',
            shape[-1],
            input_feat,
            decay=0.9)

        # Different behaviors of batch normalization
        # 1. Training, 2. Validation
        
        def _true_fn():
            # cross replica operation part
            batch_mean = tf.reduce_mean(
                input_feat,
                axis=[0, 1, 2]
                    if len(shape) == 4\
                    else [0, 1, 2, 3],
                keepdims=True)
            mean = crp_sum(batch_mean) / core_num
            var = crp_sum(
                tf.reduce_mean(
                    tf.square(
                        input_feat),
                    axis=[0, 1, 2]
                        if len(shape) == 4\
                        else [0, 1, 2, 3],
                    keepdims=True))\
                / core_num\
                - tf.square(
                    mean)

            ema_mean(mean)
            ema_mean(var)
            return mean, var
        
        def _false_fn():
            mean = tf.get_variable(
                'moving_mean',
                initializer=lambda: tf.constant(
                    0.0,
                    shape=shape),
                trainable=False)
            
            var = tf.get_variable(
                'moving_variance',
                initializer=lambda: tf.constant(
                    1.0,
                    shape=shape),
                trainable=False)

            return mean, var

        mean, var = tf.contrib.framework.smart_cond(
            tf.constant(
                is_training,
                shape=[],
                dtype=tf.bool),
            _true_fn,
            _false_fn)

        normed = tf.nn.batch_normalization(
            input_feat,
            mean,
            var,
            beta,
            gamma,
            epsilon)

    return normed


