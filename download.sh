#!/usr/bin/env bash

function fetch() {
    mkdir -p data/prefix_$1_dataset
    wget https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/dev -O data/prefix_$1_dataset/dev
    wget https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/id_to_word -O data/prefix_$1_dataset/id_to_word
    wget https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/test -O data/prefix_$1_dataset/test
    wget https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/train -O data/prefix_$1_dataset/train
    wget https://github.com/dalab/hyperbolic_nn/raw/master/prefix_$1_dataset/word_to_id -O data/prefix_$1_dataset/word_to_id
}

fetch 10
fetch 30
fetch 50
