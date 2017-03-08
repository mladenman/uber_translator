#!/usr/bin/env bash

raw_data_dir=~/uber_translator/experiments/newstest_basic/raw_data/de-en
# data_dir_subwords=~/uber_translator/experiments/newstest_basic/sets_subwords/de-en
data_dir_words=~/uber_translator/experiments/newstest_basic/sets_words/de-en

mkdir -p ${data_dir_subwords}
mkdir -p ${data_dir_words}

# ~/uber_translator/preprocessing/prepare-data.py ${raw_data_dir}/News-Commentary11.de-en de en ${data_dir_subwords} \
# --scripts ~/uber_translator/preprocessing \
# --dev-size 10000 --test-size 10000 --verbose \
# --vocab-size 30000 --subwords --shuffle --seed 1234
# --unescape-special-chars --normalize-punk

~/uber_translator/preprocessing/prepare-data.py ${raw_data_dir}/News-Commentary11.de-en de en ${data_dir_words} \
--scripts ~/uber_translator/preprocessing \
--dev-size 10000 --test-size 10000 --verbose \
--min 2 --max 50 \
--vocab-size 30000 --shuffle --seed 1234
# --unescape-special-chars --normalize-punk