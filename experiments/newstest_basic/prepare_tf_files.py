import sys
sys.path.insert(0, '/Users/mladenman/uber_translator')

from model.data_handler import *

max_sentence_length = 50
min_sentence_length = 3

ext_enc = "de"
ext_dec = "en"

folder = "de-en"

sets = ["train", "dev", "test"]

sets_path = "/Users/mladenman/uber_translator/experiments/newstest_basic/sets_words"

# make sure it has been created
tf_data_path = "/Users/mladenman/uber_translator/experiments/newstest_basic/tf_data"

prepare_data_unidirectional(
    sets_path,
    tf_data_path,
    folder,
    sets,
    ext_enc,
    ext_dec,
    max_sentence_length,
    min_sentence_length)
