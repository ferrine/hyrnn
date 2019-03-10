#!/usr/bin/env bash

function fetch() {
    mkdir -p data/prefix_$1_dataset
    curl https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/dev > data/prefix_$1_dataset/dev
    curl https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/id_to_word > data/prefix_$1_dataset/id_to_word
    curl https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/test > data/prefix_$1_dataset/test
    curl https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/train > data/prefix_$1_dataset/train
    curl https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/word_to_id > data/prefix_$1_dataset/word_to_id
}

fetch 10
fetch 30
fetch 50
