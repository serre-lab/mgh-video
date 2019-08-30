import resize_img

import pickle
from datetime import datetime
import cv2
import os
import sys
import threading
import numpy as np
import tensorflow as tf
import argparse
import multiprocessing


def _int64_feature(value):
    '''To make tf.int64 datatype
    Args:
        value: 'Integer' to specify class id
    Returns:
        TFInt64 tensor
    '''
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[value]))


def _bytes_feature(value):
    '''To make tf.bytes datatype
    Args:
        value: 'Numpy' video clip of dtype uint8
    Returns:
        TFString tensor
    '''
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value]))

def opencv_get_clip(video_name='', video_dir='',
                    start_frame=0, end_frame=0,
                    chosen_idx=None, frames_per_clip=16,
                    frame_height=224,
                    frame_width=224, channels=3):
    '''
    Get clips from video based on frame number
    Args:
        video_name: name of the video to be processed
        video_dir: 'String' to specify where videos are present
        frame: frame that represents the last frame of the clip 
        frames_per_clip: 'Integer' to specify how many frames each
            clip will contain
        frame_height: 'Integer' to specify height of frame
        frame_width: 'Integer' to specify width of frame
        channels: 'Integer' to specify channels of frame
    Returns:
        1 'Numpy' video clips of dtype: uint8
    '''
    cap = cv2.VideoCapture(
        os.path.join(
            video_dir,
            video_name))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    chunk_index_counter = 0
    data_chunk = np.empty(
        (frames_per_clip,
        frame_height,
        frame_width,
        channels), dtype=np.uint8)

    while(cap.isOpened()):
        ret, frame_matrix = cap.read()
        if ret:
            if chunk_index_counter == frames_per_clip:
                break
            if (frame_counter >= start_frame) and \
                (frame_counter <= end_frame):
                if chosen_idx[chunk_index_counter] == frame_counter:
                    # cv2 reads frames in BGR colorspace
                    # Convert BGR to RGB
                    frame_matrix = cv2.cvtColor(
                        frame_matrix,
                        cv2.COLOR_BGR2RGB)
                    
                    # Resizing to required dimension
                    frame_matrix = resize_img.resize(
                        img=frame_matrix,
                        TARGET_HEIGHT=frame_height,
                        TARGET_WIDTH=frame_width)

                    data_chunk[chunk_index_counter, :, :, :] = frame_matrix

                    chunk_index_counter += 1
               
            elif frame_counter > end_frame:
                break
            
            frame_counter += 1
        else:
            break

    cap.release()

    return data_chunk

def feat_example(label=[], data_clip=[],
                 filename='', start_frame=0,
                 end_frame=0, transition=0):
    '''
    Define and return feature and example variables
        for TFR
    Args:
        label: 'Integer' or 'List' to mention class id
            of the clip
        left_clip: 'Numpy' video clip of dtype: uint8
        right_clip: 'Numpy' video clip of dtype: uint8
        filename: 'String' to specify name of the video
        frame: 'Integer' to specify the frame number
        transition: 'Integer' to specify if transition
            data is present in experiment
    Returns:
        Protobuf example
    '''

    # Create a feature
    feature = {
        'data/label':_bytes_feature(
            tf.compat.as_bytes(
                label.tostring()))
            if transition else _int64_feature(
                label),
        'data/chunk':_bytes_feature(
            tf.compat.as_bytes(
                data_clip.tostring())),
        'data/video_name':_bytes_feature(
            filename),
        'data/start_frame':_int64_feature(
            start_frame),
        'data/end_frame':_int64_feature(
            end_frame)}

    # Create an example protocol buffer
    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature))

    return example

def _process_clips_batch(thread_index, ranges,
                        filenames, labels,
                        num_shards, shard_dir,
                        BEHAVIORS_INDICES, phase,
                        frames_per_clip, frame_height,
                        frame_width, channels,
                        transition, video_dir,
                        sampling_style='chunks'):      # 'uniform' or 'chunks'  
    '''Processes and saves list of clips as TFRecord in 1 thread.
        Each thread produces N shards where,
            N = int(num_shards / num_threads)
        For instance, if num_shards = 128, and num_threads = 16,
        then the first thread would produce shards [0, 8)
    Args:
        thread_index: 'Integer' to represent unique batch
        ranges: 'List' of pairs of integers specifying ranges of
            each batch to analyze in parallel
        filenames: 'List' of (video_name, frame number)
        labels: 'List' of behavior labels
        num_shards: shards based on :phase
        shard_dir: 'String' to specify path where shards will
            be saved
        BEHAVIORS_INDICES: 'Dict' to specify class id for each
            behavior
        phase: 'String' to specify one of train/test/val
        frames_per_clip: 'Integer' to specify how many frames each
            clip will contain
        frame_height: 'Integer' to specify height of frame
        frame_width: 'Integer' to specify width of frame
        channels: 'Integer' to specify channels of frame
        transition: 'Integer' to specify whether experiment
            contains transition data
        video_dir: 'String' to specify path where videos
            are present
    Returns:
        None
    '''
    num_threads = len(ranges)
    num_shards_per_thread = int(num_shards / num_threads)

    shard_ranges = np.linspace(
        ranges[thread_index][0],
        ranges[thread_index][1],
        num_shards_per_thread + 1).astype(int)

    num_files_in_thread = \
        ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_thread):
        # Generate a sharded version of the filename,
        # e.g: 'train-00001-of-00010'
        shard = thread_index * num_shards_per_thread + s
        output_filename = '%s-%.5d-of-%.5d'\
            %(phase, shard, num_shards)
        output_file = os.path.join(
            shard_dir,
            output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s],
            shard_ranges[s + 1],
            dtype=int)
        for i in files_in_shard:
            video_name = filenames[i][0]
            start_frame = filenames[i][1]
            end_frame = filenames[i][2]
            label = labels[i]
            if transition:
                label = []
                for lab in label:
                    for k in BEHAVIORS_INDICES.keys():
                        if k in lab:
                            lidx = BEHAVIORS_INDICES[k]
                        label.append(lidx)
                #label = [BEHAVIORS_INDICES[lab] for lab in label]
                label = np.asarray([label], dtype=np.int32) 
            else:
                #label = BEHAVIORS_INDICES[label]
                found = False
                for k in BEHAVIORS_INDICES.keys():
                    if k in label:
                        found = True
                        lidx = BEHAVIORS_INDICES[k]
                if found:
                    label = lidx
                else:
                    continue

            if (end_frame - start_frame) < frames_per_clip:
                chosen_idx = np.arange(start_frame, end_frame+1, 1)
                # Get the clip based on frame number
                data_clip = opencv_get_clip(
                    video_name=video_name,
                    video_dir=video_dir,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    chosen_idx=chosen_idx,
                    frames_per_clip=frames_per_clip,
                    frame_height=frame_height,
                    frame_width=frame_width,
                    channels=channels)

                cond_1 = not data_clip is None

                if cond_1: 
                    # Get example protocol buffer
                    example = feat_example(
                        label=label,
                        data_clip=data_clip,
                        filename=video_name,
                        start_frame=chosen_idx[0],
                        end_frame=chosen_idx[-1],
                        transition=transition)

                    # Serialize to string and write to file
                    writer.write(
                        example.SerializeToString())
                
                    shard_counter += 1
                    counter += 1

                    if not counter % 200:
                        print('%s [thread %d]: Processed %d of %d clips in thread batch.'\
                            %(datetime.now(), thread_index, counter,\
                                num_files_in_thread))
                        sys.stdout.flush()
                else:
                    print('%s, [%d, %d], [%s] problematic.'%(video_name,\
                        start_frame, end_frame, label))
            else:
                if sampling_style == 'chunks':
                    sample_itr = (end_frame - start_frame) // frames_per_clip
                    stride = (end_frame - start_frame - frames_per_clip) // sample_itr
                    for i in range(sample_itr+1):
                        chosen_idx = np.arange(start_frame + stride * i, start_frame + frames_per_clip + stride * i)
                        # Get the clip based on frame number
                        data_clip = opencv_get_clip(
                            video_name=video_name,
                            video_dir=video_dir,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            chosen_idx=chosen_idx,
                            frames_per_clip=frames_per_clip,
                            frame_height=frame_height,
                            frame_width=frame_width,
                            channels=channels)

                        cond_1 = not data_clip is None

                        if cond_1: 
                            # Get example protocol buffer
                            example = feat_example(
                                label=label,
                                data_clip=data_clip,
                                filename=video_name,
                                start_frame=chosen_idx[0],
                                end_frame=chosen_idx[-1],
                                transition=transition)

                            # Serialize to string and write to file
                            writer.write(
                                example.SerializeToString())
                        
                            shard_counter += 1
                            counter += 1

                            if not counter % 200:
                                print('%s [thread %d]: Processed %d of %d clips in thread batch.'\
                                    %(datetime.now(), thread_index, counter,\
                                        num_files_in_thread))
                                sys.stdout.flush()
                        else:
                            print('%s, [%d, %d], [%s] problematic.'%(video_name,\
                                start_frame, end_frame, label))

                    
                elif sampling_style == 'uniform':
                    sample_itr = 0
                    stride = frames_per_clip
                    idxs = np.int32(np.round(np.linspace(start_frame, end_frame, frames_per_clip)))
                    chosen_idx = idxs
                    # Get the clip based on frame number
                    data_clip = opencv_get_clip(
                        video_name=video_name,
                        video_dir=video_dir,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        chosen_idx=chosen_idx,
                        frames_per_clip=frames_per_clip,
                        frame_height=frame_height,
                        frame_width=frame_width,
                        channels=channels)

                    cond_1 = not data_clip is None

                    if cond_1: 
                        # Get example protocol buffer
                        example = feat_example(
                            label=label,
                            data_clip=data_clip,
                            filename=video_name,
                            start_frame=chosen_idx[0],
                            end_frame=chosen_idx[-1],
                            transition=transition)

                        # Serialize to string and write to file
                        writer.write(
                            example.SerializeToString())
                    
                        shard_counter += 1
                        counter += 1

                        if not counter % 200:
                            print('%s [thread %d]: Processed %d of %d clips in thread batch.'\
                                %(datetime.now(), thread_index, counter,\
                                    num_files_in_thread))
                            sys.stdout.flush()
                    else:
                        print('%s, [%d, %d], [%s] problematic.'%(video_name,\
                            start_frame, end_frame, label))

        try:
            writer.close()
            print('%s [thread %d]: Wrote %d clips to %s'\
                %(datetime.now(), thread_index, shard_counter,\
                    output_file))
            sys.stdout.flush()
            shard_counter = 0
        except:
            print('Error while writing %s'%output_file)

    print('%s [thread %d]: Wrote %d clips to %d shards.'\
        %(datetime.now(), thread_index, counter,\
            num_files_in_thread))
    sys.stdout.flush()


def _process_clips(filenames=[], labels=[], num_shards=512,
                    shard_dir='', num_threads=1,
                    BEHAVIORS_INDICES=[], phase='',
                    frames_per_clip=16, frame_height=224,
                    frame_width=224, channels=3,
                    transition=0, video_dir=''):
    '''Process and save list of clips as TFRecord of Example
        protos
    Args:
        filenames: list of (video_name, frame number)
        labels: list of behavior labels
        num_shards: shards based on :phase
        shard_dir: 'String' to specify where shards will
            be saved
        num_threads: 'Integer' to specify how many threads to
            use to parallely create shards
        BEHAVIORS_INDICES: 'Dict' to specify class id for each
            behavior
        phase: 'String' to specify one of train/test/val
        frames_per_clip: 'Integer' to specify how many frames each
            clip will contain
        frame_height: 'Integer' to specify height of frame
        frame_width: 'Integer' to specify width of frame
        channels: 'Integer' to specify channels of frame
        transition: 'Integer' to specify whether experiment
            contains transition data
        video_dir: 'String' to specify path where videos are
            present
    Returns:
        'None'
    '''

    # Break all clips into batches with a 
    # [ranges[i][0], ranges[i][1]].
    # The total number of batches equal the number of threads
    spacing = np.linspace(
        0,
        len(filenames),
        num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch   
    print('Launching %d threads for spacings: %s'\
        %(num_threads, ranges))
    sys.stdout.flush()

    if num_threads == 1:
        _process_clips_batch(
            0, ranges, filenames,
            labels, num_shards,
            shard_dir, BEHAVIORS_INDICES,
            phase, frames_per_clip,
            frame_height, frame_width,
            channels, transition,
            video_dir)
    else: 
        # Create a mechanism for monitoring when all threads finish
        coord = tf.train.Coordinator()
        for thread_index in range(len(ranges)):
            args = (thread_index,
                ranges, filenames,
                labels, num_shards,
                shard_dir, BEHAVIORS_INDICES,
                phase, frames_per_clip,
                frame_height, frame_width,
                channels, transition,
                video_dir)
            t = threading.Thread(
                target=_process_clips_batch,
                args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate
        coord.join(threads)
    
    print('%s: Finished writing all %d images in data set.'\
        %(datetime.now(), len(filenames)))
    sys.stdout.flush()


def control(args=None):
    '''Boundary function to control child modules of
        this script
    Args:
        args: 'ArgumentParser' object
    Returns:
        'None' if arguments are correct else
        raises appropriate errors
    '''

    '''
    BEHAVIORS_INDICES = {
        'adjust_items_body_L': 0,
        'adjust_items_body_R': 1,
        'adjust_items_face_or_head_L': 2,
        'adjust_items_face_or_head_R': 3,
        'background': 4,
        'finemanipulate_object': 5,
        'grasp_and_move_L': 6,
        'grasp_and_move_R': 7,
        'reach_face_or_head_L': 8,
        'reach_face_or_head_R': 9,
        'reach_nearobject_L': 10,
        'reach_nearobject_R': 11,
        'rest': 12,
        'withdraw_reach_gesture_L': 13,
        'withdraw_reach_gesture_R': 14
    }
    '''

    BEHAVIORS_INDICES = {
        'adjust_on_body_L': 0,
        'adjust_on_body_R': 1,
        'adjust_on_face_or_head_L': 2,
        'adjust_on_face_or_head_R': 3,
        'adjust_on_object_L': 4,
        'adjust_on_object_R': 5,
        'background': 6,
        'finemanipulate_object': 7,
        'grasp_and_move_object_L': 8,
        'grasp_and_move_object_R': 9,
        'reach_for_body_L': 10,
        'reach_for_body_R': 11,
        'reach_for_face_or_head_L': 12,
        'reach_for_face_or_head_R': 13,
        'reach_for_object_L': 14,
        'reach_for_object_R': 15,
        'rest_watching': 16,
        'withdraw_reach_gesture_L': 17,
        'withdraw_reach_gesture_R': 18
    }
    
    '''
    BEHAVIORS_INDICES_COMBINED = {
        'adjust_items_body': 0,
        'adjust_items_face_or_head': 1,
        'background': 2,
        'finemanipulate_object': 3,
        'grasp_and_move': 4,
        'reach_face_or_head': 5,
        'reach_nearobject': 6,
        'withdraw_reach_gesture': 7,
        #'rest': 7,
        #'withdraw_reach_gesture': 8,
    }
    '''

    BEHAVIORS_INDICES_COMBINED = {
        'adjust_on_body': 0,
        'adjust_on_face_or_head': 1,
        'adjust_on_object': 2,
        'background': 3,
        'finemanipulate_object': 4,
        'grasp_and_move_object': 5,
        'reach_for_body': 6,
        'reach_for_face_or_head': 7,
        'reach_for_object': 8,
        'rest_watching': 9,
        'withdraw_reach_gesture': 10
    }
   
    '''
    BEHAVIORS_INDICES_BINARY = {
        'reach_face_or_head': 0,
        'reach_nearobject': 1,
    }
    '''

    ##### DIRECTORIES OF INTEREST
    BASE_DIR = '/media/data_cifs/MGH/'
    EXP_NAME = 'v2_selected_pretrainedi3d_chunks_32seq_combined'
    PICKLE_SPLIT = 'v2_selected'
    PICKLE_DIR = os.path.join(BASE_DIR, 'pickle_files', PICKLE_SPLIT)
    VIDEO_DIR = os.path.join(
        BASE_DIR,
        'videos/')
    SHARDS_DIR = os.path.join(
        BASE_DIR,
        'tf_records',
        EXP_NAME,
        args.phase + '_directory/')

    # Check if directories are present
    if not os.path.exists(PICKLE_DIR):
        raise IOError('{} does not exist, create\
            meta pickle data with suitable\
            commands?'.format(PICKLE_DIR))
    pickle_filepath = os.path.join(PICKLE_DIR, args.phase + '.pkl')
    if not os.path.isfile(pickle_filepath):
        raise IOError('{}.pkl does not exist in\
            {}'.format(args.phase, PICKLE_DIR))
    if not os.path.exists(VIDEO_DIR):
        raise IOError('{} does not exist'.format(
            VIDEO_DIR))
    else:
        print('Found {} videos in {}'.format(
            len(os.listdir(VIDEO_DIR)), VIDEO_DIR))
    if os.path.exists(SHARDS_DIR):
        raise IOError('{} already exists'.format(
            SHARDS_DIR))
    else:
        os.makedirs(SHARDS_DIR)
    
    if args.multithread:
        num_threads = multiprocessing.cpu_count()
    else:
        num_threads = 1

    with open(pickle_filepath, 'rb') as handle:
        fi = pickle.load(handle)

    # Get filenames and labels 
    np.random.shuffle(fi)
    filenames, labels = [], []
    for li in fi:
        filenames.append(li[0])
        labels.append(li[1])

    _process_clips(
        filenames=filenames,
        labels=labels,
        num_shards=args.num_shards,
        shard_dir=SHARDS_DIR,
        num_threads=num_threads,
        BEHAVIORS_INDICES=BEHAVIORS_INDICES_COMBINED,
        #BEHAVIORS_INDICES=BEHAVIORS_INDICES_BINARY,
        phase=args.phase,
        frames_per_clip=args.frames_per_clip,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        channels=args.channels,
        transition=args.transition,
        video_dir=VIDEO_DIR)


    
def check_arguments(args=None):
    '''To check if arguments are correct
    Args:
        args: 'ArgumentParser' object
    Returns:
        'None' if arguments are correct else
        raises appropriate errors
    '''
    frames_per_clip = args.frames_per_clip
    phase = args.phase
    num_shards = args.num_shards
    frame_height = args.frame_height
    frame_width = args.frame_width
    channels = args.channels
    multithread = args.multithread
    transition = args.transition
    clips_per_behavior = args.clips_per_behavior

    assert type(frames_per_clip) == int, '\'frames_per_clip\'\
        should be of type Integer, found {}'.format(type(frames_per_clip))
    assert type(phase) == str, '\'phase\' should be of type\
        String, found {}'.format(type(phase))
    cond_1 = phase == 'train'
    cond_2 = phase == 'val'
    cond_3 = phase == 'test'
    assert type(num_shards) == int, '\'num_shards\' should be\
        of type Integer, found {}'.format(type(num_shards))
    assert type(frame_height) == int, '\'frame_height\' should\
        be of type Integer, found {}'.format(type(frame_height))
    assert type(frame_width) == int, '\'frame_width\' should\
        be of type Integer, found {}'.format(type(frame_width))
    assert type(channels) == int, '\'channels\' should\
        be of type Integer, found {}'.format(type(channels))
    assert type(multithread) == int, '\'multithread\' should\
        be of type Integer, found {}'.format(type(multithread))
    assert type(transition) == int, '\'transition\' should be\
        of type Integer, found {}'.format(type(transition))
    assert type(clips_per_behavior) == int, '\'clips_per_behavior\'\
        should be of type Integer, found {}'.format(type(clips_per_behavior)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='params of running the experiment')

    parser.add_argument(
        '--frames_per_clip',
        type=int,
        default=32)

    parser.add_argument(
        '--phase',
        type=str,
        default='train')

    parser.add_argument(
        '--num_shards',
        type=int,
        default=512)

    parser.add_argument(
        '--frame_height',
        type=int,
        default=224)

    parser.add_argument(
        '--frame_width',
        type=int,
        default=224)

    parser.add_argument(
        '--channels',
        type=int,
        default=3)

    parser.add_argument(
        '--multithread',
        type=int,
        default=1)

    parser.add_argument(
        '--transition',
        type=int,
        default=0)

    parser.add_argument(
        '--clips_per_behavior',
        type=int,
        default=10000)

    args = parser.parse_args()
    check_arguments(args)
    control(args)
