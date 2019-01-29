import logging
import tensorflow as tf
import timeline
from seq2seq_model_fn import *
from seq2seq_input_fn import *

seq2seq_train.py
def train_seq2seq(
        input_filename,
        output_filename,
        vocab_filename,
        model_dir):

    vocab = load_vocab(vocab_filename)
    params = {
        'vocab_size': len(vocab),
        'batch_size': 32,
        'input_max_length': 30,
        'output_max_length': 30,
        'embed_dim': 100,
        'num_units': 256
    }
    est = tf.estimator.Estimator(
        model_fn=seq2seq,
        model_dir=model_dir, params=params)

    input_fn, feed_fn = make_input_fn(
        params['batch_size'],
        input_filename,
        output_filename,
        vocab, params['input_max_length'], params['output_max_length'])

    # Make hooks to print examples of inputs/predictions.
    print_inputs = tf.train.LoggingTensorHook(
        ['input_0', 'output_0'], every_n_iter=100,
        formatter=get_formatter(['input_0', 'output_0'], vocab))
    print_predictions = tf.train.LoggingTensorHook(
        ['predictions', 'train_pred'], every_n_iter=100,
        formatter=get_formatter(['predictions', 'train_pred'], vocab))

    timeline_hook = timeline.TimelineHook(model_dir, every_n_iter=100)
    est.train(
        input_fn=input_fn,
        hooks=[tf.train.FeedFnHook(feed_fn), print_inputs, print_predictions,
               timeline_hook],
        steps=10000)


def main():
    tf.logging._logger.setLevel(logging.INFO)
    train_seq2seq('input', 'output', 'vocab', 'model/seq2seq')


if __name__ == "__main__":
    main()

