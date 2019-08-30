from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import time
from absl import flags
import absl.logging as _logging
from absl import app
import numpy as np
import pickle

# central reservoir classes
import inp_pipeline
from central_reservoir.models import i3d
from central_reservoir.utils.layers import avgpool
from central_reservoir.utils.layers import conv_batchnorm_relu

from tensorflow.python.estimator import estimator
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.contrib import summary

#TODO: 1. 5-D Augmentation scheme, 2. Document structure,
# 3. Fix bfloat16 issue.

#BEHAVIORS_INDICES = {
#    'adjust_items_body_L': 0,
#    'adjust_items_body_R': 1,
#    'adjust_items_face_or_head_L': 2,
#    'adjust_items_face_or_head_R': 3,
#    'background': 4,
#    'finemanipulate_object': 5,
#    'grasp_and_move_L': 6,
#    'grasp_and_move_R': 7,
#    'reach_face_or_head_L': 8,
#    'reach_face_or_head_R': 9,
#    'reach_nearobject_L': 10,
#    'reach_nearobject_R': 11,
#    'rest': 12,
#    'withdraw_reach_gesture_L': 13,
#    'withdraw_reach_gesture_R': 14
#}

BEHAVIORS_INDICES = {
    'adjust_items_body': 0,
    'adjust_items_face_or_head': 1,
    'background': 2,
    'finemanipulate_object': 3,
    'grasp_and_move': 4,
    'reach_face_or_head': 5,
    'reach_nearobject': 6,
    'rest': 7,
    'withdraw_reach_gesture': 8
}

#class_weights = {
# 'adjust_items_body_L': 43.0,
# 'adjust_items_body_R': 41.0,
# 'adjust_items_face_or_head_L': 44.0,
# 'adjust_items_face_or_head_R': 42.0,
# 'background': 90.0,
# 'finemanipulate_object': 26.0,
# 'grasp_and_move_L': 15.0,
# 'grasp_and_move_R': 19.0,
# 'reach_face_or_head_L': 52.0,
# 'reach_face_or_head_R': 41.0,
# 'reach_nearobject_L': 24.0,
# 'reach_nearobject_R': 45.0,
# 'rest': 48.0,
# 'withdraw_reach_gesture_L': 57.0,
# 'withdraw_reach_gesture_R': 72.0
#}

class_weights = {
 'adjust_items_body': 84.0,
 'adjust_items_face_or_head': 86.0,
 'background': 90.0,
 'finemanipulate_object': 26.0,
 'grasp_and_move': 34.0,
 'reach_face_or_head': 93.0,
 'reach_nearobject': 69.0,
 'rest': 48.0,
 'withdraw_reach_gesture': 129.0,
}

FLAGS = flags.FLAGS

##### Cloud TPU Cluster Resolvers
flags.DEFINE_bool(
    'use_tpu',
    default=True,
    help='Use TPU to execute model. If False,'
            'use whatever devices are available by'
            'default: TPU / GPU')

flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should'
            'be the name of the TPU or the ip address of it')

flags.DEFINE_string(
    'eval_tpu',
    default=None,
    help='The Cloud TPU to be used for evaluating. Use this '
            'if FLAGS.mode == train_and_eval, and '
            'FLAGS.train_num_cores > 8')

flags.DEFINE_string(
    'gcp_project',
    default='beyond-dl-1503610372419',
    help='Project name for the Cloud-enabled TPU project')

flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in.')

flags.DEFINE_string(
    'eval_tpu_zone',
    default='us-central1-f',
    help='GCE zone where the Cloud TPU is located in.')

##### Data directory flags
flags.DEFINE_string(
    'data_dir',
    default=None,
    help='The directory where the dataset is present')

flags.DEFINE_string(
    'model_dir',
    default=None,
    help='The directory where the model and train/eval'
            'summaries are stored')

##### Model specific flags
flags.DEFINE_string(
    'final_endpoint',
    default='Logits',
    help='Which endpoint to stop at when constructing the network graph')

flags.DEFINE_integer(
    'time_divisor',
    default=4,
    help='Subsample T dimension from the output of endpoint in network to feed to Logits')

flags.DEFINE_integer(
    'hw_divisor',
    default=12,
    help='Subsample H,W dimension from the output of endpoint in network to feed to Logits')

flags.DEFINE_string(
    'warm_start_vars',
    default='Conv3d_*w, Conv3d_*beta, Mixed_3*w, Mixed_3*beta, Mixed_4*w, Mixed_4*beta, Mixed_5*w, Mixed_5*beta',
    help='List of REs that help to warm start the network weights during initialization')

flags.DEFINE_string(
    'optimize_var_scopes',
    default='Mixed_5b,Logits',
    help='List of REs that help to warm start the network weights during initialization')

flags.DEFINE_integer(
    'profile_every_n_steps',
    default=0,
    help='Number of steps between collecting profiles if'
            'larger than 0')

flags.DEFINE_bool(
    'double_logits',
    default=False,
    help='Create model with two FC layers')

flags.DEFINE_bool(
    'use_batch_norm',
    default=False,
    help='Create model with/without batch norm layers')

flags.DEFINE_bool(
    'use_cross_replica_batch_norm',
    default=False,
    help='Boolean to specify if cross shard BN is to be done')

flags.DEFINE_string(
    'mode',
    default='train',
    help='One of train_and_eval, train, evaluate')

flags.DEFINE_integer(
    'train_steps',
    default=112590,
    help='The number of steps to use for training')

flags.DEFINE_integer(
    'train_batch_size',
    default=512,
    help='Total batch size for training')

flags.DEFINE_integer(
    'eval_batch_size',
    default=24,
    help='Total batch size for evaluation')

flags.DEFINE_integer(
    'predict_batch_size',
    default=32,
    help='Total batch size for prediction')

flags.DEFINE_integer(
    'num_train_videos',
    default=9000,
    help='Total training video clips')

flags.DEFINE_integer(
    'num_eval_videos',
    default=2000,
    help='Total evaluation video clips')

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before '
        'evaluation terminates')

flags.DEFINE_integer(
    'num_classes',
    default=9,
    help='Number of classes '
        '9 for mice, 101 for UCF')

flags.DEFINE_integer(
    'steps_per_eval',
    default=1251,
    help='Controls how often evaluation is done')

flags.DEFINE_bool(
    'skip_host_call',
    default=False,
    help='Skip the host_call when executed training step.'
            'Generally used for generating training'
            'summaries')

flags.DEFINE_integer(
    'iterations_per_loop',
    default=1251,
    help='Number of steps to run on TPU before outfeeding'
            'metrics to the CPU. If the number of iterations '
            'in the loop exceeds the number of train steps, '
            'the loop will exit before reaching this flag. '
            'The larger this value is, the higher the '
            'utilization on the TPU')

##### tf.data.Dataset related flags
flags.DEFINE_integer(
    'num_parallel_calls',
    default=8,
    help='Number of parallel threads in CPU for the input'
            'pipeline. Recommended value is the number of cores '
            'per CPU host')

flags.DEFINE_integer(
    'train_num_cores',
    default=8,
    help='Number of TPU cores in total. Options: 8/32/64/128/'
            '256/512')

flags.DEFINE_integer(
    'eval_num_cores',
    default=8,
    help='Numer of TPU cores in total. Right now, only v2-8 '
            'can be used for inferencing')

flags.DEFINE_bool(
    'use_cache',
    default=False,
    help='Enable cache for training input')

flags.DEFINE_string(
    'class_num_samples',
    default='/media/data_cifs/MGH/pickle_files/v1_selected/class_weights.pkl',
    help='Path to pickle file containing the number of samples per class')

##### Training related flags
flags.DEFINE_string(
    'export_dir',
    default=None,
    help='The directory where the exported SavedModel will be'
            'stored')

flags.DEFINE_bool(
    'export_to_tpu',
    default=False,
    help='Whether to export additional metagraph')

flags.DEFINE_string(
    'precision',
    default='bfloat16',
    help='Precision to use: bfloat16, float32')

flags.DEFINE_string(
    'optimizer',
    default='sgd',
    help='Specify optimizer: [adam, sgd]')

flags.DEFINE_string(
    'init_checkpoint',
    default='gs://serrelab/biomotion/checkpoints/model.ckpt',
    help='The checkpoint from which you want to initialize the weights')

flags.DEFINE_float(
    'base_learning_rate',
    default=0.01,
    help='Base learning rate when tran batch size is 256')

flags.DEFINE_float(
    'momentum',
    default=0.9,
    help='Momentum parameter used in Momentum optimizer')

flags.DEFINE_float(
    'weight_decay',
    default=1e-4,
    help='Weight decay coefficient for l2 reg')

flags.DEFINE_float(
    'label_smoothing',
    default=0.0,
    help='Label smoothing parameter used in the '
            'softmax cross entropy')

flags.DEFINE_integer(
    'log_step_count_steps',
    default=64,
    help='The number of steps at which the global step '
            'information is logged')

flags.DEFINE_bool(
    'enable_lars',
    default=False,
    help='Enable LARS optimizer for large batch training')

flags.DEFINE_bool(
    'use_async_checkpointing',
    default=False,
    help='Enable async checkpointing') 


def get_lr_schedule(train_steps, num_train_videos, train_batch_size):
    '''Learning rate schedule'''
    steps_per_epoch = np.floor(
        num_train_videos / train_batch_size)
    train_epochs = train_steps / steps_per_epoch
    return [ # (multiplier, epoch to start) tuples
        (1.0, np.floor(5 / 90 * train_epochs)),
        (0.1, np.floor(30 / 90 * train_epochs)),
        (0.01, np.floor(60 / 90 * train_epochs)),
        (0.001, np.floor(80 / 90 * train_epochs))
    ]


def learning_rate_schedule(train_steps, current_epoch):
    '''Handles linear scaling rule, gradual warmup, and LR decay.
    The learning rate starts at 0, increases linearly per step.
    After 5 epochs, the base learning rate is reached.
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training spots and the LR is set to 0.

    Args:
        1. train_steps: int number of training steps
        2. current_epoch: Tensor for the current epoch

    Returns:
        Scaled Tensor for current epochs
    '''
    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 128.0) #256.0)

    lr_schedule = get_lr_schedule(
        train_steps=train_steps,
        num_train_videos=FLAGS.num_train_videos,
        train_batch_size=FLAGS.train_batch_size)
    decay_rate = (scaled_lr * lr_schedule[0][0] *
        current_epoch / lr_schedule[0][1])
  
    for mult, start_epoch in lr_schedule:
        decay_rate = tf.where(
            current_epoch < start_epoch,
            decay_rate, scaled_lr * mult)
  
    return decay_rate


def i3d_model_fn(features, labels, mode, params):
    '''The model_fn for I3D to be used with TPUEstimator.
    
    Args:
        1. features: 'Tensor' of batched images
        2. labels: 'Tensor' of labels
        3. mode: one of 'tf.estimator.ModeKeys.\
            {TRAIN, EVAL, PREDICT}'
        4. params: 'dict' of parameters passed by TPUEstimator.
            params['batch_size'] is always provided and should be
                used as effective batch size.

    Returns:
        A TPUEstimatorSpec for the model
    '''

    def build_network():
        network = i3d.InceptionI3d(
            final_endpoint=FLAGS.final_endpoint,
            use_batch_norm=FLAGS.use_batch_norm,
            use_cross_replica_batch_norm=FLAGS.use_cross_replica_batch_norm,
            num_cores=FLAGS.train_num_cores,
            num_classes=FLAGS.num_classes,
            spatial_squeeze=True,
            dropout_keep_prob=0.7)

        if FLAGS.final_endpoint == 'Logits':
            logits, end_points = network(
                inputs=features['video'],
                is_training=(mode == tf.estimator.ModeKeys.TRAIN))

            return logits
        else:
            descriptors, end_points = network(
                inputs=features['video'],
                is_training=(mode == tf.estimator.ModeKeys.TRAIN))
            # From the descriptors, we need to downsample now. That should be an input argument.
            t_subsample_factor = FLAGS.time_divisor
            hw_subsample_factor = FLAGS.hw_divisor
            spatial_squeeze = True
            end_point = 'Logits'
            with tf.variable_scope(end_point):
                fc_inputs = avgpool(descriptors, ksize=[1, t_subsample_factor, hw_subsample_factor, hw_subsample_factor, 1],
                        strides=[1, 1, 1, 1, 1], padding='VALID')
                get_shape = fc_inputs.get_shape().as_list()
                print('{} / Average-pool3D: {}'.format(end_point, get_shape))
                end_points[end_point + '_average_pool3d'] = fc_inputs

                # Dropout
                fc_inputs = tf.nn.dropout(fc_inputs, 0.7)

                # Use two FC layers
                if FLAGS.double_logits:
                    fc_inputs = conv_batchnorm_relu(fc_inputs, 'Conv3d_xx_1x1', 256,
                            kernel_size=1, stride=1, use_batch_norm=FLAGS.use_batch_norm,
                            use_cross_replica_batch_norm=FLAGS.use_cross_replica_batch_norm,
                            is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                            num_cores=FLAGS.train_num_cores)
                    get_shape = fc_inputs.get_shape().as_list()
                    print('{} / Conv3d_xx_1x1 : {}'.format(end_point, get_shape))
            
                # 1x1x1 Conv, stride 1
                logits = conv_batchnorm_relu(fc_inputs, 'Conv3d_0c_1x1', FLAGS.num_classes,
                    kernel_size=1, stride=1, activation=None,
                    use_batch_norm=False, use_cross_replica_batch_norm=False,
                    is_training=(mode == tf.estimator.ModeKeys.TRAIN), num_cores=FLAGS.train_num_cores)
                get_shape = logits.get_shape().as_list()
                print('{} / Conv3d_0c_1x1 : {}'.format(end_point, get_shape))

		if spatial_squeeze:
                    # Removes dimensions of size 1 from the shape of a tensor
                    # Specify which dimensions have to be removed: 2 and 3
                    logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
                    get_shape = logits.get_shape().as_list()
                    print('{} / Spatial Squeeze : {}'.format(end_point, get_shape))

            averaged_logits = tf.reduce_mean(logits, axis=1) # [N, num_classes]
            get_shape = averaged_logits.get_shape().as_list()
            print('{} / Averaged Logits : {}'.format(end_point, get_shape))

            end_points[end_point] = averaged_logits
            return averaged_logits

    # Speed up computation, saves memory
    if FLAGS.precision == 'bfloat16':
        with tf.contrib.tpu.bfloat16_scope():
            logits = build_network()

        logits = tf.cast(
            logits,
            tf.float32)
    elif FLAGS.precision == 'float32':
        logits = build_network()

    # Don't know the use of this yet
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits_argmax = tf.argmax(logits, axis=-1)
        oh_labels = tf.one_hot(labels, FLAGS.num_classes, dtype=tf.float32)
        labels_argmax = tf.argmax(oh_labels, axis=-1)
        predictions = {
            'ground_truth': labels_argmax,
            'predictions': logits_argmax}
        
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions)

    # :batch_size is per_core batch size. Estimator has taken care of this
    # in main() already
    batch_size = params['batch_size']
    one_hot_labels = tf.one_hot(
        labels,
        FLAGS.num_classes,
        dtype=tf.float32)
    
    # Scale the cross entropy by the weight value for that class (handling imbalance)
    cls_numbers = class_weights
    cls_weights = []
    tot = 0
    for k,v in cls_numbers.items():
        tot += v
    for k,v in sorted(cls_numbers.items()):
        cls_weights.append(tot/v)
    #print(cls_weights)
    weights = tf.constant(cls_weights)
    weights_to_apply = tf.gather(weights, labels)

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
        label_smoothing=FLAGS.label_smoothing,
        weights=weights_to_apply)

    # Add weight decay to the loss of for non-batch normalization variables.
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'batch_normalization' not in v.name])
    
    # Calculate training accuracy.
    # labels : <batch_size, 1>
    # one_hot_labels: <batch_size, num_classes>
    # logits: <batch_size>, <num_classes>
    # train_predictions: <batch_size>
    train_predictions = tf.argmax(
        logits,
        axis=1)
    ground_truth = tf.argmax(
        one_hot_labels,
        axis=1)
    correct_train_pred = tf.equal(
        train_predictions,
        ground_truth)
    train_1_accuracy = tf.reduce_mean(
        tf.cast(
            correct_train_pred,
            tf.float32))
    
    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from a 
        # global step
        global_step = tf.train.get_global_step()
        steps_per_epoch = FLAGS.num_train_videos / FLAGS.train_batch_size
        current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)

        var_scopes = FLAGS.optimize_var_scopes.split(',')
        var_scopes = [x.strip() for x in var_scopes]
        if len(var_scopes) > 0:
            vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scopes[0])
            for sc in var_scopes[1:]:
                vars_to_optimize += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
        else:
            vars_to_optimize = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # LARS is a large batch optimizer. LARS enables higher accuracy at batch
        # 16K and larger batch sizes.
        if FLAGS.train_batch_size >= 16384 and FLAGS.enable_lars:
            learning_rate = 0.0
            optimizer = lars_util.init_lars_optimizer(
                current_epoch)
        else:
            if FLAGS.optimizer == 'sgd':
                # Following schedule used in training Imagenet.
                #learning_rate = learning_rate_schedule(
                #    FLAGS.train_steps,
                #    current_epoch)
                # Exponential decay learning rate
            
                learning_rate = tf.train.exponential_decay(
                    FLAGS.base_learning_rate,
                    global_step,
                    300,
                    0.96,
                    staircase=True)
                #learning_rate = tf.constant(0.0, tf.float32)
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate,
                    momentum=FLAGS.momentum,
                    use_nesterov=True)

            elif FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=FLAGS.base_learning_rate)

        if FLAGS.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(
                optimizer)
        
        if FLAGS.use_batch_norm:
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss,
                    global_step,
                    var_list=vars_to_optimize)
        else:
            train_op = optimizer.minimize(
                loss,
                global_step,
                var_list=vars_to_optimize)
        
        if not FLAGS.skip_host_call:
            def host_call_fn(gs, loss, acc, ce):#lr, ce):
                '''Training host call. Creates scalar summaries for training
                    metrics.
                    This funciton is executed on the CPU. As in, after 
                    :iterations_per_loop computation in TPU, control moves to
                    the CPU where the summaries are updated.
                    Arguments should match the list of 'Tensor' objects passed as
                    the second element in the tuple passed to 'host_call'.
                Args:
                    gs: Tensor with shape [batch] for global step
                    loss: Tensor with shape [batch] for the training loss
                    lr: Tensor with shape [batch] for the learning rate
                    ce: Tensor with shape [batch] for the current epoch

                Returns:
                    List of summary ops to run on the CPU host.
                '''
                gs = gs[0]
                # Host call fns are executed FLAGS.iterations_per_loop times after
                # one TPU loop is finished, setting max_queue value to the same as
                # number of iterations will make the summary writer only flush the
                # data to storage once per loop.
                with summary.create_file_writer(
                    FLAGS.model_dir,
                    max_queue=FLAGS.iterations_per_loop).as_default():

                    with summary.always_record_summaries():
                        summary.scalar('loss', loss[0], step=gs)
                        summary.scalar('top_1', acc[0], step=gs)
                        #summary.scalar('top_5', t5_acc[0], step=gs)
                        #summary.scalar('learning_rate', lr[0], step=gs)
                        summary.scalar('current_epoch', ce[0], step=gs)

                        return summary.all_summary_ops()
            
            # To log the metrics, the summary op needs to be run on the host CPU
            # via host_call. host_call expects [batch_size, ...] Tensors, thus
            # reshape to introduce a batch dimension. These Tensors are implicitly
            # concatenated to [params['batch_size']].
            gs_t = tf.reshape(global_step, [1])
            loss_t = tf.reshape(loss, [1])
            top_1_acc_t = tf.reshape(train_1_accuracy, [1])
            #top_5_acc_t = tf.reshape(top_5_training_accuracy, [1])
            #lr_t = tf.reshape(learning_rate, [1])
            ce_t = tf.reshape(current_epoch, [1])

            host_call = (
                host_call_fn,
                [gs_t, loss_t, top_1_acc_t, ce_t])#lr_t, ce_t])

    else:
        train_op = None

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, logits):
            '''Evaluation metric function. Evaluates accuracy.
            This function is executed on the CPU.

            Args:
                labels: 'Tensor' with shape [batch]
                logits: 'Tensor' with shape [batch, num_classes]

            Returns:
                A dict of the metrics to return from evaluation.
            '''
            predictions = tf.argmax(
                logits,
                axis=1)
            top_1_accuracy = tf.metrics.accuracy(
                labels,
                predictions)

            return {
                'top_1_accuracy': top_1_accuracy,}

        # If other variables are to be used for evaluation purposes, get them
        # using tf.global_variables() and add them to the list of variables, such
        # as [labels, logits, var_1, ...., var_n]. Similarly, in metric_fn, define
        # parameters to take represent these extra variables.
        eval_metrics = (metric_fn, [labels, logits])

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=eval_metrics)


def main(unused_argv):

    # Check flag conditions:
    if FLAGS.mode == 'train':
        tf.logging.info(
            'Mode = train, TPU = %s, Num cores = %d'
            %(FLAGS.tpu, FLAGS.train_num_cores))

    elif FLAGS.mode == 'evaluate':
        tf.logging.info(
            'Mode = evaluate, TPU = %s, Num cores = %d'
            %(FLAGS.eval_tpu, FLAGS.eval_num_cores))

    elif FLAGS.mode == 'train_and_eval':
        if FLAGS.train_num_cores > 8:
            tf.logging.info(
                'Mode = train_and_eval, Train TPU = %s, '
                'Train num cores: %d, Eval TPU = %s, '
                'Eval num cores: %d'
                %(FLAGS.tpu, FLAGS.train_num_cores,
                FLAGS.eval_tpu, FLAGS.eval_num_cores))
        else:
            tf.logging.info(
                'Mode = train_and_eval, TPU = %s, '
                'Num cores: %d'
                %(FLAGS.tpu, FLAGS.train_num_cores))

    # Set up general purpose tpu_cluster_resolver based on FLAGS.mode:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu
            if FLAGS.mode in ['train', 'train_and_eval']
            else FLAGS.eval_tpu,
        zone=FLAGS.tpu_zone
            if FLAGS.mode in ['train', 'train_and_eval']
            else FLAGS.eval_tpu_zone,
        project=FLAGS.gcp_project)
    
    # For mode == 'train_and_eval' we can have 2 options:
    # 1. Use same TPU for training and evaluating (only v2-8)
    # 2. Use TPU with more cores for training (v2-32/128/256/512),
    #       and a separate v2-8 for evaluating.
    if FLAGS.mode == 'train_and_eval' and FLAGS.train_num_cores > 8:
        eval_tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.eval_tpu,
            zone=FLAGS.eval_tpu_zone,
            project=FLAGS.gcp_project)  
       
    if FLAGS.use_async_checkpointing:
        save_checkpoints_steps = None
    else:
        save_checkpoints_steps = max(
            100,
            FLAGS.iterations_per_loop)

    ##### RunConfig parameters:
    '''Arguments:
        iterations_per_loop: number of training steps running in TPU system
            before returning to CPU host for each Session.run. Global step is
            increased iterations_per_loop times in one Session.run. It is recommended
            to be set as number of global steps for next checkpoint.
        per_host_input_for_training: If True, input_fn is invoked once on each host.
            If PER_HOST_V1: batch size per shard = train_batch_size // #hosts (#cpus)
            If PER_HOST_V2: batch size per shard = train_batch_size // #cores  
        keep_checkpoint_max: If None, keep all checkpoint files, otherwise specify
            'n' to keep latest 'n' files.

    Each TPU device has 8 cores and is connected to a host (CPU). Larger slices have
    multiple hosts. For instance, v2-256 communicates with 16 hosts. So, per_host_input_\
    for_training will invoke/create the Dataset pipeline 16 times in total for 16 hosts,
    where each host will serve 256/16 = 16 cores. Each core will take a batch size represented
    by flag PER_HOST_V2. This functionality is missing right now in tf.Keras which makes it
    difficult to scale up models to bigger TPU slices.

    '''
    config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=FLAGS.log_step_count_steps,
        keep_checkpoint_max=None,
        session_config=tf.ConfigProto(
            graph_options=tf.GraphOptions(
                rewrite_options=rewriter_config_pb2.RewriterConfig(
                    disable_meta_optimizer=True))),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.train_num_cores
                if FLAGS.mode in ['train', 'train_and_eval']
                else FLAGS.eval_num_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.\
                PER_HOST_V2))

    if FLAGS.mode == 'train_and_eval' and FLAGS.train_num_cores > 8:
        config_eval = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver_eval,
            model_dir=FLAGS.model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            log_step_count_steps=FLAGS.log_step_count_steps,
            keep_checkpoint_max=None,
            session_config=tf.ConfigProto(
                graph_options=tf.GraphOptions(
                    rewrite_options=rewriter_config_pb2.RewriterConfig(
                        disable_meta_optimizer=True))),
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.eval_num_cores,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.\
                    PER_HOST_V2))

   
    ##### Estimator story:
    '''Estimator handles running details, such as replicating inputs and models for
        core, and returning to host periodically to run hooks.
        -> TPUEstimator transforms a global batch size in params to a per-shard/core
            batch size when calling input_fn and model_fn. Users SHOULD specify GLOBAL
            batch size in constructor and then get the batch size for EACH shard/core 
            in input_fn and model_fn by PARAMS['BATCH_SIZE'].
        -> For training, model_fn gets per_core_batch_size; input_fn may get
            per-core or per-host batch size depending on per_host_input_for_training in
            TPUConfig. For this model, we use PER_HOST_V2.
        -> For evaluation and prediction, model_fn gets per-core batch size and input_fn
            per-host batch size.

        Current limitations:
            -> TPU prediction only works on a single host (one TPU worker)
            -> input_fn must return a Dataset instance rather than features. In fact,
                train(), and evaluate() also support Dataset as return value.
    '''
    '''Arguments:
        model_fn: Should be a TPUEstimatorSpec. 
        use_tpu: Setting to False for testing. All training, evaluation, and predict will
            be executed on CPU. input_fn and model_fn will receive train_batch_size or
            eval_batch_size unmodified as params['batch_size']. Setting to True, input_fn
            and model_fn will receive per_core batch size. :config plays a role in specifying
            details about TPU workers to the Estimator.
        config: An tpu_config.RunConfig configuration object. Cannot be None.
        params: An optional dict of hyper parameters that will be passed into input_fn and
            model_fn. Keys are names of parameters, values are basic python types. There are
            reserved keys for TPUEstimator, including 'batch_size'. Extra parameters can be 
            added to this dictionary and can be used in input_fn and model_fn scripts.
        train_batch_size: An int representing the global batch size. TPUEstimator transforms
            this global batch size to a per-shard/core batch size, as params['batch_size'],
            when calling input_fn and model_fn. Cannot be None if :use_tpu is True. Must be
            DIVISIBLE by total number of replicas. The per-shard batch size calculation is
            automatically done using TPUConfig details.
        export_to_tpu: If True, export_savedmodel() exports a metagraph for serving on TPU
            besides the one on CPU.
    '''
    
    if not FLAGS.init_checkpoint == 'None':
        warm_start_vars = FLAGS.warm_start_vars.split(',')
        warm_start_vars = [x.strip() for x in warm_start_vars]
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.init_checkpoint,
            vars_to_warm_start=warm_start_vars
        )

        i3d_classifier = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=i3d_model_fn,
            config=config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            export_to_tpu=FLAGS.export_to_tpu,
            warm_start_from=ws)
    else:
        i3d_classifier = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=i3d_model_fn,
            config=config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            export_to_tpu=FLAGS.export_to_tpu)

    if FLAGS.mode == 'train_and_eval' and FLAGS.train_num_cores > 8:
        i3d_eval = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=i3d_model_fn,
            config=config_eval,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            export_to_tpu=FLAGS.export_to_tpu,
            warm_start_from=ws)

    assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
        'Invalid value for --precision flag; must be bfloat16 or float32.')
    tf.logging.info(
        'Precision: %s',
        FLAGS.precision)

    use_bfloat16 = FLAGS.precision == 'bfloat16'

    tf.logging.info(
        'Using dataset: %s',
        FLAGS.data_dir)
    
    list_of_augmentations = [
        'random_crop',
        'random_brightness',
        'random_contrast']

    # dataset_train and dataset_eval are the Input pipelines
    dataset_train, dataset_eval, dataset_predict = [
        inp_pipeline.InputPipelineTFExample(
            data_dir=FLAGS.data_dir,
            is_training=is_training,
            cache=FLAGS.use_cache and is_training,
            use_bfloat16=use_bfloat16,
            target_image_size=224,
            num_frames=32,      # num_frames_change_here
            num_classes=15,
            num_parallel_calls=FLAGS.num_parallel_calls,
            list_of_augmentations=list_of_augmentations)
        for is_training in [True, False, False]]

    # num_train_videos = total images in the dataset
    # train_batch_size = total batch size (across all cores)
    steps_per_epoch = FLAGS.num_train_videos // FLAGS.train_batch_size
    eval_steps = FLAGS.num_eval_videos // FLAGS.eval_batch_size

    if FLAGS.mode == 'train' or FLAGS.mode == 'evaluate':
        
        # Automatically get the latest checkpoint file and latest
        # train step from the model_dir.
        current_step = estimator._load_global_step_from_checkpoint_dir(
            FLAGS.model_dir)
        
        tf.logging.info(
            'Training for %d steps (%.2f epochs in total). Current'
            'step %d.',
            FLAGS.train_steps,
            FLAGS.train_steps / steps_per_epoch,
            current_step)

        start_timestamp = time.time() # Compilation time included

        if FLAGS.mode == 'train':
            hooks = []
            
            # Not sure what this does. I think this takes care of
            # asynchronously saving checkpoint files, irrespective of
            # training routine on TPU.
            if FLAGS.use_async_checkpointing:
                hooks.append(
                    async_checkpoint.AsyncCheckpointSaverHook(
                        checkpoint_dir=FLAGS.model_dir,
                        save_steps=max(100, FLAGS.iterations_per_loop)))

            # Number of steps between collecting prog=files if larger
            # than 0.
            if FLAGS.profile_every_n_steps > 0:
                hooks.append(
                    tpu_profiler_hook.TPUProfilerHook(
                        save_steps=FLAGS.profile_every_n_steps,
                        output_dir=FLAGS.model_dir,
                        tpu=FLAGS.tpu))

            ##### Estimator training story:
            '''Arguments:
                input_fn: Returns mini batches for training. Function should
                    return tf.data.Dataset object: tuple (features, labels).
                    Both features and labels are consumed by model_fn. They
                    should satisfy the expectation of model_fn for inputs.
                hooks: List of tf.train.SessionRunHook subclass instance. Used
                    for callbacks inside the training loop.
                max_steps: Number of total steps for which to train the model.
            '''
            i3d_classifier.train(
                input_fn=dataset_train.input_fn,
                max_steps=FLAGS.train_steps,
                hooks=hooks)

        elif FLAGS.mode == 'evaluate':
            '''
            for ckpt in evaluation.checkpoints_iterator(
                FLAGS.model_dir, timeout=FLAGS.eval_timeout):
                tf.logging.info(
                    'Starting to evaluate using %s',
                    ckpt)
            '''
            f = open('evaluations/dummy_' + FLAGS.model_dir.split('/')[-1] + '.txt', 'ab')
            #ids = [i for i in range(12600, 14000, 300)]
            #ids.append(14000)
            ids = [14000]
            #import ipdb; ipdb.set_trace()
            for i in ids:
                try:
                    ckpt = FLAGS.model_dir + '/model.ckpt-' + str(i)
                    start_timestamp = time.time() # Compilation time included
                    eval_results = i3d_classifier.evaluate(
                        input_fn=dataset_eval.input_fn,
                        steps=eval_steps,
                        checkpoint_path=ckpt)
                    elapsed_time = int(time.time() - start_timestamp)
                    tf.logging.info(
                        'Eval results: %s. Elapsed seconds: %d',
                        eval_results,
                        elapsed_time)

                    f.write('step: ' + str(i) + ', stats: ' + str(eval_results) + '\n')
                    f.close()
                    f = open('evaluations/dummy_' + FLAGS.model_dir.split('/')[-1] + '.txt', 'ab')

                    # Terminate eval job when final checkpoint is reached
                    current_step = int(os.path.basename(ckpt).split('-')[1])
                    if current_step >= FLAGS.train_steps:
                        tf.logging.info(
                            'Evaluation finished after training step %d',
                            current_step)
                        break

                except tf.errors.NotFoundError:
                    tf.logging.info(
                        'Checkpoint %s no longer exists, skipping checkpoint',
                        ckpt)
            f.close()
    
    elif FLAGS.mode == 'predict':
        i = 1000
        ckpt = FLAGS.model_dir + '/model.ckpt-' + str(i)
        predict_iters = i3d_classifier.predict(
            input_fn=dataset_predict.input_fn,
            checkpoint_path=ckpt,
            yield_single_examples=False)
        all_gt, all_preds = [], []
        count = 0
        for predict_result in predict_iters:
            gt = predict_result['ground_truth']
            preds = predict_result['predictions']
            if count % 10 == 0:
                print('step:{}, shapes:{}'.format(count, gt.shape))
            count += 1

            for j in gt:
                all_gt.append(j)
                all_preds.append(j)

        print('Finished, {}'.format(len(all_gt)))
        with open('gt.pkl', 'wb') as handle:
            pickle.dump(all_gt, handle)
        with open('preds.pkl', 'wb') as handle:
            pickle.dump(all_preds, handle)

if __name__ == '__main__':
    tf.logging.set_verbosity(
        tf.logging.INFO)
    app.run(main)
