#!/bin/bash

set -x

# source /Users/a58/Documents/code/nanochat/.venv/bin/activate
# cd /Users/a58/Documents/code/autograd
# python test_train_gpt2.py
# deactivate

rm -rf build
rm metal_ext.cpython-310-darwin.so

/Users/a58/Documents/code/nanochat/.venv/bin/python setup.py build_ext --inplace
