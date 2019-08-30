import random
import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from central_reservoir.models import i3d
from central_reservoir.augmentations import preprocessing_volume

from absl import flags
from absl import app

# Get test shards from bucket
test_shards_path = '/media/data_cifs/MGH/tf_records/v1_selected_pretrainedi3d_chunks_32seq_combined/mgh_test_directory'
shards = tf.gfile.Glob(
    os.path.join(
        test_shards_path,
        'mgh_test*'))
print('{} testing shards found'.format(len(shards)))
test_examples = 200
batch_size = 20
print('Testing examples: {}'.format(test_examples))

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_folder_name',
    default='v1_chunks_32seq_combined_v3-8_b256_9classes_adamlre-3_i3d_weightedloss_earlyendpoint_block+logits',
    help='To mention the model path')

flags.DEFINE_integer(
    'step',
    default=14000,
    help='To specify the checkpoint')

'''
BEHAVIOR_INDICES = {
    0: 'adjust_items_body_L',
    1: 'adjust_items_body_R',
    2: 'adjust_items_face_or_head_L',
    3: 'adjust_items_face_or_head_R',
    4: 'background',
    5: 'finemanipulate_object',
    6: 'grasp_and_move_L',
    7: 'grasp_and_move_R',
    8: 'reach_face_or_head_L',
    9: 'reach_face_or_head_R',
    10: 'reach_nearobject_L',
    11: 'reach_nearobject_R',
    12: 'rest',
    13: 'withdraw_reach_gesture_L',
    14: 'withdraw_reach_gesture_R'
}

behaviors = [
    'adjust_items_body_L',
    'adjust_items_body_R',
    'adjust_items_face_or_head_L',
    'adjust_items_face_or_head_R',
    'background',
    'finemanipulate_object',
    'grasp_and_move_L',
    'grasp_and_move_R',
    'reach_face_or_head_L',
    'reach_face_or_head_R',
    'reach_nearobject_L',
    'reach_nearobject_R',
    'rest',
    'withdraw_reach_gesture_L',
    'withdraw_reach_gesture_R'
]
'''

BEHAVIOR_INDICES = {
    0: 'adjust_items_body',
    1: 'adjust_items_face_or_head',
    2: 'background',
    3: 'finemanipulate_object',
    4: 'grasp_and_move',
    5: 'reach_face_or_head',
    6: 'reach_nearobject',
    7: 'rest',
    8: 'withdraw_reach_gesture',
}

behaviors = [
    'adjust_items_body',
    'adjust_items_face_or_head',
    'background',
    'finemanipulate_object',
    'grasp_and_move',
    'reach_face_or_head',
    'reach_nearobject',
    'rest',
    'withdraw_reach_gesture',
]

slack_window = 20
each_side = slack_window / 2

def read_pickle(fi):
    with open(fi, 'rb') as handle:
        a = pickle.load(handle)
    return a

def temporal(fnames, sframes, eframes, top_class_batch, labels_batch):
    temporal_top_class_batch = []
    for i in range(len(fnames)):
        all_frames = videoname_frame[fnames[i]]
        frame_in_focus = fnumbers[i]

        get_window = all_frames[
            frame_in_focus - each_side :\
            frame_in_focus + each_side + 1]

        behav_id_in_focus = top_class_batch[i]
        behav_in_focus = BEHAVIOR_INDICES[behav_id_in_focus]

        if behav_in_focus in get_window:
            temporal_top_class_batch.append(labels_batch[i])
        else:
            temporal_top_class_batch.append(top_class_batch[i])

    return temporal_top_class_batch


def plot_confusion_matrix(png_path='',
            cnf_matrix=None,
            classes=[],
            annot_1='Ground_truth',
            annot_2='Predictions',
            balanced_acc=0.0,
            title='',
            cmap=plt.cm.Blues):

    plt.figure(figsize=(18,16))
    plt.imshow(
        cnf_matrix,
        interpolation='nearest',
        cmap=cmap)

    title = title + ', Balanced acc: {}'.format(round(balanced_acc))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(cnf_matrix))

    plt.xticks(
        tick_marks,
        classes,
        rotation=45,
        fontsize=8)
    plt.yticks(
        tick_marks,
        classes,
        fontsize=8)

    fmt = '.2f'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                horizontalalignment='center',
                color='white' if cnf_matrix[i, j] > thresh else 'black')

    plt.ylabel(annot_1)
    plt.xlabel(annot_2)
    plt.tight_layout()

    plt.savefig(png_path)

def get_bal_acc(matrix):
    norm_matrix = np.zeros(
        (
            len(matrix),
            len(matrix)),
        dtype=np.float32)

    for i in range(len(matrix)):
        get_li = matrix[i].astype('float')
        sum_li = sum(get_li)
        if sum_li != 0:
            norm_matrix[i] = get_li / sum_li
        else:
            norm_matrix[i] = [0.0 for i in range(len(matrix))]

    diagonal, good_classes = 0, 0
    for i in range(len(norm_matrix)):
        good_classes += 1
        diagonal += norm_matrix[i][i]

    bal_acc = (diagonal / float(good_classes)) * 100.0

    return bal_acc, norm_matrix

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, ser = reader.read(filename_queue)

    keys_to_features = {
        'data/chunk': tf.FixedLenFeature(
            [],
            tf.string),
        'data/label': tf.FixedLenFeature(
            [],
            tf.int64),
        'data/video_name': tf.FixedLenFeature(
            [],
            tf.string),
        'data/start_frame': tf.FixedLenFeature(
            [],
            tf.int64),
        'data/end_frame': tf.FixedLenFeature(
            [],
            tf.int64)}

    parsed = tf.parse_single_example(
        ser,
        features=keys_to_features)

    video = tf.decode_raw(
        parsed['data/chunk'],
        tf.uint8)

    label = tf.cast(
        parsed['data/label'],
        tf.int32)

    video_name = parsed['data/video_name']

    start_frame = tf.cast(
        parsed['data/start_frame'],
        tf.int32)

    end_frame = tf.cast(
        parsed['data/end_frame'],
        tf.int32)

    height, width = 256, 256

    video = preprocessing_volume.preprocess_volume(
        volume=video,
        num_frames=32,
        height=height,
        width=width,
        is_training=False,
        target_image_size=224,
        use_bfloat16=False,
        list_of_augmentations=['random_crop'])

    videos, labels, video_names, sframes, eframes = tf.train.batch([video, label, video_name, start_frame, end_frame],
        batch_size=batch_size,
        capacity=30,
        num_threads=1)

    return videos, labels, video_names, sframes, eframes

def main(unused_argv):

    all_preds, all_ground = [], []

    common_path = '/media/data_cifs/MGH/model_runs/'
    ckpt_path = os.path.join(
        common_path,
        FLAGS.model_folder_name,
        'model.ckpt-{}'.format(FLAGS.step))

    with tf.Session().as_default() as sess:
        filename_queue = tf.train.string_input_producer(
            shards,
            num_epochs=1)
        print(filename_queue)
        video, label, filename, sframes, eframes = read_and_decode(filename_queue)
        #label = tf.one_hot(label, 15, dtype=tf.float32)
        label = tf.one_hot(label, 9, dtype=tf.float32)

        network = i3d.InceptionI3d(
            final_endpoint='Logits',
            use_batch_norm=True,
            use_cross_replica_batch_norm=True,
            num_classes=9,
            spatial_squeeze=True,
            dropout_keep_prob=1.0)

        logits, end_points = network(
            inputs=video,
            is_training=False)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            coord=coord)

        try:
            for i in range(int(test_examples/batch_size)):
                preds, labs, fname, sframe, eframe = sess.run(
                    [logits, label, filename, sframes, eframes])
                print(fname)
                preds_max = list(np.argmax(preds, axis=-1))
                labs_max = list(np.argmax(labs, axis=-1))
                all_preds += preds_max
                all_ground += labs_max

                #temporal_top_class_batch = temporal(
                #    fname,
                #    snumber,
                #    enumber,
                #    preds_max,
                #    labs_max)

                #all_temporal_preds += temporal_top_class_batch

                if  ( ((i+1)*batch_size) % 50 == 0):
                    print('{}/{} completed'.format((i+1)*batch_size, test_examples))

            coord.request_stop()

        except:
            print('{} examples covered'.format(i))

    print('Testing complete')

    ground_behav = [BEHAVIOR_INDICES[i] for i in all_ground]
    preds_behav = [BEHAVIOR_INDICES[i] for i in all_preds]
    #temporal_preds_behav = [BEHAVIOR_INDICES[i] for i in all_temporal_preds]

    with open('gt.pkl', 'wb') as handle:
        pickle.dump(ground_behav, handle)
    with open('preds.pkl', 'wb') as handle:
        pickle.dump(preds_behav, handle)
    #with open('temporal_preds.pkl', 'wb') as handle:
    #    pickle.dump(temporal_preds_behav, handle)

    cnf_matrix_ground_preds = confusion_matrix(ground_behav, preds_behav)
    #cnf_matrix_ground_temporal_preds = confusion_matrix(ground_behav, temporal_preds_behav)
    np.set_printoptions(precision=2)
    gp_balacc, gp_normmatrix = get_bal_acc(cnf_matrix_ground_preds)
    #gtp_balacc, gtp_normmatrix = get_bal_acc(cnf_matrix_ground_temporal_preds)

    plot_confusion_matrix(png_path='/home/kalpitthakkar/v1_selected_ground_preds.png',
        cnf_matrix=gp_normmatrix,
        classes=behaviors,
        annot_1='Ground_truth',
        annot_2='Predictions',
        balanced_acc=gp_balacc,
        title='v1_selected_labels_test_performance: ',
        cmap=plt.cm.Blues)

    #plot_confusion_matrix(png_path='/home/kalpitthakkar/v1_selected_ground_temporal_preds_slack20.png',
    #    cnf_matrix=gtp_normmatrix,
    #    classes=behaviors,
    #    annot_1='Ground_truth',
    #    annot_2='Predictions',
    #    balanced_acc=gtp_balacc,
    #    title='v1_selected_labels_test_performance, slack=20, : ',
    #    cmap=plt.cm.Blues)


if __name__ == '__main__':
    app.run(main)
