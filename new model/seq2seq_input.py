PATH_UBER_TRANSLATOR = '/Users/mladenman/uber_translator'
PATH_WMT2014_DATA_RAW = '/Users/mladenman/uber_translator/experiments/wmt2014/raw_data'
PATH_WMT2014_DATA_TF = '/Users/mladenman/uber_translator/experiments/wmt2014/tf_data'


import multiprocessing
import sys
sys.path.insert(0, PATH_UBER_TRANSLATOR)

from model.data_handler import *


vocab_en, rev_vocab_en = initialize_vocabulary(os.path.join(PATH_WMT2014_DATA_RAW, 'wmt2014.vocab.50K.en'))
vocab_de, rev_vocab_de = initialize_vocabulary(os.path.join(PATH_WMT2014_DATA_RAW, 'wmt2014.vocab.50K.de'))

prepare_data_unidirectional(
    PATH_WMT2014_DATA_RAW,
    PATH_WMT2014_DATA_TF,
    'test',
    'en',
    'de',
    100,
    5
)

print('prepare test data done')

prepare_data_unidirectional(
    PATH_WMT2014_DATA_RAW,
    PATH_WMT2014_DATA_TF,
    'train',
    'en',
    'de',
    100,
    5
)

print('prepare train data done')

prepare_data_unidirectional(
    PATH_WMT2014_DATA_RAW,
    PATH_WMT2014_DATA_TF,
    'val',
    'en',
    'de',
    100,
    5
)

print('prepare val data done')

#load np files
idx_q_train = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_q.train.npy'))
idx_a_train = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_a.train.npy'))

idx_q_val = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_q.val.npy'))
idx_a_val = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_a.val.npy'))

idx_q_test = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_q.test.npy'))
idx_a_test = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_a.test.npy'))



tfrecord_test = os.path.join(PATH_WMT2014_DATA_TF, 'train.tfrecord')
tfrecord_train = os.path.join(PATH_WMT2014_DATA_TF, 'test.tfrecord')
tfrecord_val = os.path.join(PATH_WMT2014_DATA_TF, 'val.tfrecord')

with open(tfrecord_train, 'w') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
for i in range(0, len(tfrecord_train)):
    record = sequence_to_tf_example(sequence_en=idx_q_train[i], sequence_de=idx_a_train[i])
    writer.write(record.SerializeToString())

print('prepare tfrecord train done')

with open(tfrecord_test, 'w') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
for i in range(0, len(tfrecord_test)):
    record = sequence_to_tf_example(sequence_en=idx_q_test[i], sequence_de=idx_a_test[i])
    writer.write(record.SerializeToString())

print('prepare tfrecord test done')


class ReadTFRecords(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, glob_pattern,training=True):
        """
        Read tf records matching a glob pattern
        :param glob_pattern:
            glob pattern eg. "/usr/local/share/Datasets/Imagenet/train-*.tfrecords"
        :param training:
            Whether or not to shuffle the data for training and evaluation
        :return:
            Iterator generating one example of batch size for each training step
        """
        threads = multiprocessing.cpu_count()
        with tf.name_scope("tf_record_reader"):
            # generate file list
            files = tf.data.Dataset.list_files(glob_pattern, shuffle=training)

            # parallel fetch tfrecords dataset using the file list in parallel
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename), cycle_length=threads))

            # shuffle and repeat examples for better randomness and allow training beyond one epoch
            dataset = dataset.shuffle(32*self.batch_size)

            # map the parse  function to each example individually in threads*2 parallel calls
            dataset = dataset.map(map_func=lambda example: parse(example),
                                  num_parallel_calls=threads)

            # batch the examples
            dataset = dataset.batch(batch_size=self.batch_size)

            #prefetch batch
            dataset = dataset.prefetch(buffer_size=32)

        return dataset.make_one_shot_iterator().get_next()
