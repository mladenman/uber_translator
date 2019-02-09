import os
import numpy as np
import pickle
from random import sample
import tensorflow as tf


def decode(sequence, lookup, separator=''):  # 0 used for padding, is ignored
    return separator.join([lookup[element] for element in sequence if element])


def batch_gen(x, y, batch_size):  # ne treba mi u novoj
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i: (i + 1) * batch_size].T, y[i: (i+1) * batch_size].T


def rand_batch_gen(x, y, batch_size):  # ne treba mi u novoj
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T


def prepare_data_unidirectional(
        #  kreira numpy i metadata dumpove podataka, uzima vocab i OCISCENE set podatke
        #
        sets_path,
        tf_data_path,
        dataset_name,
        subset,
        ext_enc,
        ext_dec,
        max_sentence_length,
        min_sentence_length):

    path = os.path.join(sets_path)

    vocab_file_questions = os.path.join(path, "wmt2014.vocab.50K.{}".format(ext_enc))
    vocab_file_answers = os.path.join(path, "wmt2014.vocab.50K.{}".format(ext_dec))

    w2idx_questions, idx2w_questions = initialize_vocabulary(vocab_file_questions)
    w2idx_answers, idx2w_answers = initialize_vocabulary(vocab_file_answers)

    words_questions, words_answers = read_words(path, dataset_name, subset, ext_enc, ext_dec)

    idx_questions, idx_answers = word2idx(
                                    words_questions,
                                    words_answers,
                                    w2idx_questions,
                                    w2idx_answers,
                                    max_sentence_length,
                                    min_sentence_length)

    np.save(os.path.join(tf_data_path, 'idx_q.{}.npy'.format(subset)), idx_questions)
    np.save(os.path.join(tf_data_path, 'idx_a.{}.npy'.format(subset)), idx_answers)

    metadata = {
        'enc_input_length': max_sentence_length,
        'dec_input_length': max_sentence_length,
        'w2idx_enc': w2idx_questions,
        'w2idx_dec': w2idx_answers,
        'idx2w_enc': idx2w_questions,
        'idx2w_dec': idx2w_answers
    }

    if not os.path.exists(os.path.join(tf_data_path, 'metadata.pkl')):
        with open(os.path.join(tf_data_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f, 2)


# FUNCTIONS FOR BIDIRECTIONAL TRANSLATOR - SHOULD BE DONE AS CASE OF MULTIDIRECTIONAL
#
# def prepare_data_bidirectional(data_path, subsets, ext1, ext2):
#    for subset in subsets:
#        words_questions, words_answers = read_words(data_path, subset, ext1, ext2)
#        words_questions, words_answers = merge(words_questions, words_answers, ext1, ext2)
#        idx_questions, idx_answers, w2idx, idx2w = word2idx_merged(words_questions, words_answers)
#
#        np.save('idx_q.npy', idx_questions)
#        np.save('idx_a.npy', idx_answers)
#        metadata = {
#            'w2idx': w2idx,
#            'idx2w': idx2w
#        }
#        with open('metadata.pkl', 'wb') as f:
#            pickle.dump(metadata, f)
#
# def merge(words_questions, words_answers, ext1, ext2):
#    pass
#
# def word2idx_merged(words_questions, words_answers):
#    pass


def read_words(path, dataset_name, subset, ext1, ext2):

    file_questions = os.path.join(path, "{}.{}.{}".format(dataset_name, subset, ext1))
    file_answers = os.path.join(path, "{}.{}.{}".format(dataset_name, subset, ext2))

    lines_questions = open(file_questions, encoding='utf-8', errors='ignore').read().split('\n')
    lines_answers = open(file_answers, encoding='utf-8', errors='ignore').read().split('\n')

    words_questions = [[w.strip() for w in sentence.split(' ') if w] for sentence in lines_questions]
    words_answers = [[w.strip() for w in sentence.split(' ') if w] for sentence in lines_answers]

    return words_questions, words_answers


def word2idx(words_questions, words_answers, w2idx_questions, w2idx_answers, max_sentence_length, min_sentence_length):

    idx_questions = pad(words_questions, w2idx_questions, max_sentence_length, min_sentence_length)
    idx_answers = pad(words_answers, w2idx_answers, max_sentence_length, min_sentence_length)

    return idx_questions, idx_answers


def pad(words, w2idx, max_sentence_length, min_sentence_length):

    data_len = len(words)
    idx_q = np.zeros([data_len, max_sentence_length], dtype=np.int32)
    real_set_size = 0

    for i in range(data_len):

        q_indices = pad_seq(words[i], w2idx, max_sentence_length)
        if max_sentence_length >= len(q_indices) >= min_sentence_length:
            idx_q[real_set_size] = np.array(q_indices + [0] * (max_sentence_length - len(q_indices))) # morao da vratim pad zbog numpyja
            real_set_size += 1

    print(idx_q[:real_set_size].shape)

    return idx_q[:real_set_size]


def pad_seq(seq, lookup, max_sentence_length):

    indices = []

    for word in seq:

        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup['<unk>'])

    return indices # + [lookup['<PAD>']] * (max_sentence_length - len(seq)) ako hou da vratim pad nekad


def initialize_vocabulary(vocabulary_path):
    """
    Returns 2 dictionaries needed for transforming idx -> words and vice versa
    :param vocabulary_path:
    :return vocab, rev_vocab:
    """
    rev_vocab = []

    with open(vocabulary_path) as f:
        rev_vocab.extend(f.readlines())

    rev_vocab = [line.rstrip('\n') for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

    return vocab, rev_vocab


def load_tf_data(path):

    with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    idx_q_train = np.load(os.path.join(path, 'idx_q.train.npy'))
    idx_a_train = np.load(os.path.join(path, 'idx_a.train.npy'))

    idx_q_dev = np.load(os.path.join(path, 'idx_q.dev.npy'))
    idx_a_dev = np.load(os.path.join(path, 'idx_a.dev.npy'))

    idx_q_test = np.load(os.path.join(path, 'idx_q.test.npy'))
    idx_a_test = np.load(os.path.join(path, 'idx_a.test.npy'))

    return metadata, idx_q_train, idx_a_train, idx_q_dev, idx_a_dev, idx_q_test, idx_a_test


def sequence_to_tf_example(sequence_en, sequence_de):
    ex = tf.train.SequenceExample()

    fl_tokens_en = ex.feature_lists.feature_list["tokens_en"]
    fl_tokens_en.feature.add().int64_list.value.append(2)   # vocab['<s>']
    for token in sequence_en:
        fl_tokens_en.feature.add().int64_list.value.append(token)
    fl_tokens_en.feature.add().int64_list.value.append(3)   # vocab['<\s>']

    fl_tokens_de = ex.feature_lists.feature_list["tokens_de"]
    fl_tokens_de.feature.add().int64_list.value.append(2)   # vocab['<s>']
    for token in sequence_de:
        fl_tokens_de.feature.add().int64_list.value.append(token)
    fl_tokens_de.feature.add().int64_list.value.append(3)   # vocab['<\s>']

    return ex


def parse(ex):
    '''
    Explain to TF how to go from a serialized example back to tensors
    :param ex:
    :return:
    '''
    sequence_features = {
        "tokens_en": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "tokens_de": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # Parse the example (returns a dictionary of tensors)
    sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        sequence_features=sequence_features
    )

    return {"seq_en": sequence_parsed["tokens_en"], "seq_de": sequence_parsed["tokens_de"]}
