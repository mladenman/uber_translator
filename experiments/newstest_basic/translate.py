import sys
sys.path.insert(0, '/Users/mladenman/uber_translator')


from model.translator import Translator
from model.data_handler import *


tf_files_path = '/Users/mladenman/uber_translator/experiments/newstest_basic/tf_data/'
checkpoints_path = '/Users/mladenman/uber_translator/experiments/newstest_basic/checkpoints/'


# load data from pickle and npy files
metadata, \
    idx_q_train, \
    idx_a_train, \
    idx_q_dev, \
    idx_a_dev, \
    idx_q_test, \
    idx_a_test = load_tf_data(tf_files_path)


# parameters
xseq_len = metadata['enc_input_length']  # trainX.shape[-1]
yseq_len = metadata['dec_input_length']  # trainY.shape[-1]

xvocab_size = len(metadata['idx2w_enc'])
yvocab_size = len(metadata['idx2w_dec'])

batch_size = 256
emb_dim = 512
num_layers = 3
layer_size = 512
test_batch_size = 16

model = Translator(xseq_len=xseq_len,
                   yseq_len=yseq_len,
                   xvocab_size=xvocab_size,
                   yvocab_size=yvocab_size,
                   emb_dim=emb_dim,
                   num_layers=num_layers,
                   layer_size=layer_size,
                   ckpt_path=checkpoints_path
                   )

# TRAINING
# dev_batch_gen = rand_batch_gen(idx_q_dev, idx_a_dev, batch_size)
# train_batch_gen = rand_batch_gen(idx_q_train, idx_a_train, batch_size)
#
# sess = model.train(train_batch_gen, dev_batch_gen)

# TESTING
sess = model.restore_last_session()

test_batch_gen = rand_batch_gen(idx_q_train, idx_a_train, test_batch_size)

input_ = test_batch_gen.__next__()[0]

output = model.predict(sess, input_)
# print(output.shape)
for ii, oi in zip(input_.T, output):

    q = decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')

    decoded = decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')

    print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))

