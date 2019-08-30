import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#mgh_path = '/media/data_cifs/MGH/tf_records/v2_selected_pretrainedi3d_chunks_32seq_combined/mgh_train_directory/'
mgh_path = '/media/data_cifs/MGH/tf_records/v2_selected_pretrainedi3d_chunks_32seq_combined/mgh_test_directory/'
mgh_shards = [mgh_path + fi for fi in os.listdir(mgh_path)]

def mgh_read(f):
    queue = tf.train.string_input_producer(f, num_epochs=1)
    reader = tf.TFRecordReader()
    _, ser = reader.read(queue)

    feature = {
        'data/label': tf.FixedLenFeature([], tf.int64),
        'data/chunk': tf.FixedLenFeature([], tf.string),
        'data/video_name': tf.FixedLenFeature([], tf.string),
        'data/start_frame': tf.FixedLenFeature([], tf.int64),
        'data/end_frame': tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_single_example(ser, features=feature)
    data_clip = tf.decode_raw(parsed['data/chunk'], tf.uint8)
    label = tf.cast(parsed['data/label'], tf.int32)
    video_name = parsed['data/video_name']
    start = tf.cast(parsed['data/start_frame'], tf.int32)
    end = tf.cast(parsed['data/end_frame'], tf.int32)

    data_clip = tf.reshape(data_clip, [32, 256, 256, 3])
    #data_clip = tf.reshape(data_clip, [16, 256, 256, 3])
    data_clips, labels, vidname = tf.train.shuffle_batch(
        [data_clip, label, video_name],
        batch_size=1,
        capacity=100,
        min_after_dequeue=50,
        num_threads=2)

    return data_clips, labels, vidname, start, end

lc, ls, vname, s, e = mgh_read(mgh_shards)

init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    ctr = 0
    threads = tf.train.start_queue_runners(coord=coord)
    print(len(mgh_shards))
    for i in range(380):
        l, la, vn, st, en = sess.run([lc, ls, vname, s, e])
        import ipdb; ipdb.set_trace()
        ctr += 1
    print ctr
    coord.request_stop()
    coord.join(threads)
    sess.close()
