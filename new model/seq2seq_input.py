

from datetime import datetime


PATH_UBER_TRANSLATOR = '/Users/mladenman/uber_translator'
PATH_WMT2014_DATA_RAW = '/Users/mladenman/uber_translator/experiments/wmt2014/raw_data'
PATH_WMT2014_DATA_TF = '/Users/mladenman/uber_translator/experiments/wmt2014/tf_data'
DATASET_NAME_TRAIN = 'wmt2014'
DATASET_NAME_TEST = 'newstest2014'


import multiprocessing
import sys
sys.path.insert(0, PATH_UBER_TRANSLATOR)

from model.data_handler import *


vocab_en, rev_vocab_en = initialize_vocabulary(os.path.join(PATH_WMT2014_DATA_RAW, 'wmt2014.vocab.50K.en'))
vocab_de, rev_vocab_de = initialize_vocabulary(os.path.join(PATH_WMT2014_DATA_RAW, 'wmt2014.vocab.50K.de'))

#prepare_data_unidirectional(
#    PATH_WMT2014_DATA_RAW,
#    PATH_WMT2014_DATA_TF,
#    DATASET_NAME_TEST,
#    'test',
#    'en',
#    'de',
#    200,
#    0
#)
#
#print('prepare test data done')
#
#prepare_data_unidirectional(
#    PATH_WMT2014_DATA_RAW,
#    PATH_WMT2014_DATA_TF,
#    DATASET_NAME_TRAIN,
#    'train',
#    'en',
#    'de',
#    200,
#    0
#)
#
#print('prepare train data done')
#
#prepare_data_unidirectional(
#    PATH_WMT2014_DATA_RAW,
#    PATH_WMT2014_DATA_TF,
#    DATASET_NAME_TRAIN,
#    'val',
#    'en',
#    'de',
#    200,
#    0
#)
#
#print('prepare val data done')

#load np files
idx_q_train = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_q.train.npy'))
idx_a_train = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_a.train.npy'))

idx_q_val = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_q.val.npy'))
idx_a_val = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_a.val.npy'))

idx_q_test = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_q.test.npy'))
idx_a_test = np.load(os.path.join(PATH_WMT2014_DATA_TF, 'idx_a.test.npy'))



tfrecord_test = os.path.join(PATH_WMT2014_DATA_TF, 'test.tfrecord')
tfrecord_train = os.path.join(PATH_WMT2014_DATA_TF, 'train.tfrecord')
tfrecord_val = os.path.join(PATH_WMT2014_DATA_TF, 'val.tfrecord')

startTime = datetime.now()
with open(tfrecord_val, 'w') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
for i in range(0, len(idx_q_val)):
    record = sequence_to_tf_example(sequence_en=idx_q_val[i], sequence_de=idx_a_val[i])
    writer.write(record.SerializeToString())

print('prepare tfrecord val done')
print(datetime.now() - startTime)
print()

startTime = datetime.now()
with open(tfrecord_test, 'w') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
for i in range(0, len(idx_q_test)):
    record = sequence_to_tf_example(sequence_en=idx_q_test[i], sequence_de=idx_a_test[i])
    writer.write(record.SerializeToString())

print('prepare tfrecord test done')
print(datetime.now() - startTime)
print()

startTime = datetime.now()
with open(tfrecord_train, 'w') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
for i in range(0, len(idx_q_train)):
    record = sequence_to_tf_example(sequence_en=idx_q_train[i], sequence_de=idx_a_train[i])
    writer.write(record.SerializeToString())

print('prepare tfrecord train done')
print(datetime.now() - startTime)
print()



