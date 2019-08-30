'''Implementation of cross shard
batch-normalization operation on TPUs.
Main author: Dingdong Yang
Collab: Rohit Saha
'''

import tensorflow as tf
from tensorflow.python.training import moving_averages

class ExponentialMovingAverage():
    def __init__(self,
                name, 
                decay=0.99,
                param_shape=None):

        value_initialized = 1.0 if name == 'moving_variance' else 0.0

        self.decay = decay
        self.moving_stat = tf.get_variable(
            name,
            initializer=lambda: tf.constant(
                value_initialized,
                shape=param_shape),
            trainable=False)

    def __call__(self, input_var):
        with tf.colocate_with(self.moving_stat):
            update_stat = self.moving_stat\
                            * self.decay\
                            + input_var\
                            * (1 - self.decay)

        #assign_op = tf.assign(
        #    self.moving_stat,
        #    update_stat).op
        #assign_op._unconditional_update = False

        #tf.add_to_collection(
        #    tf.GraphKeys.UPDATE_OPS,
        #    assign_op)
        tf.add_to_collection(
            'MOVING_STATS_UPDATE_OP',
            tf.assign(
                self.moving_stat,
                update_stat))

        tf.add_to_collection(
            'MOVING_STATS',
            self.moving_stat)

    def __call__(self,
                local_stat):

        update_stat = moving_averages.assign_moving_average(
            self.moving_stat,
            local_stat,
            self.decay,
            zero_debias=False)

        return update_stat
 
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
        axis, param_shape = ([0, 1, 2], [1, 1, 1, shape[-1]]) if len(shape) == 4 \
                else ([0, 1, 2, 3], [1, 1, 1, 1, shape[-1]])

        if center:
            beta = tf.get_variable(
                name='beta',
                initializer=lambda: tf.constant(
                    0.0,
                    shape=param_shape),
                trainable=True)
        else:
            beta = tf.constant(
                0.0,
                shape=param_shape,
                name='beta')

        if scale:
            gamma = tf.get_variable(
                name='gamma',
                initializer=lambda: tf.constant(
                    1.0,
                    shape=param_shape),
                trainable=True)
        else:
            gamma = tf.constant(
                1.0,
                shape=param_shape,
            name='gamma')


        # Cross replica local statistics computation
        batch_mean = tf.reduce_mean(
            input_feat,
            axis=axis,
            keepdims=True)
        mean = crp_sum(batch_mean) / core_num
        variance = crp_sum(
            tf.reduce_mean(
                tf.square(input_feat), axis=axis, keepdims=True
                )) / core_num - tf.square(mean)

        # Moving statistics adding
        ema_mean = ExponentialMovingAverage(
            'moving_mean',
            decay=0.99,
            param_shape=param_shape)
        ema_variance = ExponentialMovingAverage(
            'moving_variance',
            decay=0.99,
            param_shape=param_shape)
        _moving_vars_fn = lambda: (ema_mean.moving_stat, ema_variance.moving_stat)
        _delay_update = lambda: (ema_mean(mean), ema_variance(variance))

        update_mean, update_variance = tf.contrib.framework.smart_cond(
            tf.constant(is_training, shape=[], dtype=tf.bool),
            true_fn=_delay_update, false_fn=_moving_vars_fn)

        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            update_mean)
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            update_variance)
        #-------------------------------------------------------------------------

        _vars_fn = lambda: (mean, variance)
        mean, variance = tf.contrib.framework.smart_cond(
            tf.constant(is_training, shape=[], dtype=tf.bool),
            true_fn=_vars_fn, false_fn=_moving_vars_fn)

        normed = tf.nn.batch_normalization(
            input_feat,
            mean,
            variance,
            beta,
            gamma,
            epsilon)

        return normed

