set -x

rm -rf build metal_dw2_debug.metal
/Users/a58/Documents/code/nanochat/.venv/bin/python setup.py build_ext --inplace
