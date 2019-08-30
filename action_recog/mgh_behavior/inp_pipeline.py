import os
import tensorflow as tf

from central_reservoir.augmentations import preprocessing_volume 

class InputPipelineTFExample(object):

    def __init__(self,
                data_dir,
                is_training=True,
                cache=False,
                use_bfloat16=False,
                target_image_size=224,
                num_frames=16,
                num_classes=15,
                num_parallel_calls=8,
                list_of_augmentations=[]):

        self.data_dir = data_dir
        self.is_training = is_training
        self.cache = cache
        self.use_bfloat16 = use_bfloat16
        self.target_image_size = target_image_size
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.num_parallel_calls = num_parallel_calls
        self.image_preprocessing_fn = preprocessing_volume.preprocess_volume
        self.list_of_augmentations = list_of_augmentations

    def _dataset_parser(self, value):
        '''Parses a video chunk and its label from a serialized TFExample

        Returns a tuple of (chunk, label) from the TFExample
        '''

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
            value,
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

        video = self.image_preprocessing_fn(
            volume=video,
            num_frames=self.num_frames,
            height=height,
            width=width,
            is_training=self.is_training,
            target_image_size=self.target_image_size,
            use_bfloat16=self.use_bfloat16,
            list_of_augmentations=self.list_of_augmentations)

        # One hot conversion takes place in model_fn.

        data_dict = {
            'video': video,
            #'video_name': video_name,
            'start_frame': start_frame,
            'end_frame': end_frame
        }

        return data_dict, label

    def input_fn(self, params):
        '''Input function that provides a single batch for train/eval
    
        params: 'dict' of paramaters passed by TPUEstimator class
            params['batch_size'] is always present and should be used
        '''

        # Retrieves the batch size for the current shard
        batch_size = params['batch_size']
        
        file_pattern = os.path.join(
            self.data_dir,
            'mgh_train*' if self.is_training else 'mgh_test*')
        print(file_pattern)
        
        # For multi-host training (in the case of v2-32/64/128/512),
        # we want each host to process the same subset of shards.
        # This allows us to cache larger datasets into memory
        dataset = tf.data.Dataset.list_files(
            file_pattern,
            shuffle=False)
        
        # 'context' present when using more than 1 host, in case of
        # v2-32/64/128/256/512.
        if 'context' in params:
            current_host = params['context'].current_input_fn_deployment()[1]
            num_hosts = params['context'].num_hosts
        else:
            current_host = 0
            num_hosts = 1

        # For multi-host training, we want each host to always process
        # the same subset of files. Each host only sees a subset of the
        # entire dataset, allowing us to cache larger datasets in memory.
        dataset = dataset.shard(
            num_hosts,
            current_host) 

        if self.is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024 # 8 Mb per file
            dataset = tf.data.TFRecordDataset(
                filename,
                buffer_size=buffer_size)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                fetch_dataset,
                cycle_length=8,
                sloppy=True))

        if self.cache:
            dataset = dataset.cache().apply(
                tf.data.experimental.shuffle_and_repeat(
                    1024 * 16))
        else:
            dataset = dataset.shuffle(128)
        
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                self._dataset_parser,
                batch_size=batch_size,
                num_parallel_calls=8,
                drop_remainder=True))

        dataset = dataset.prefetch(
            tf.contrib.data.AUTOTUNE)

        return dataset
    
