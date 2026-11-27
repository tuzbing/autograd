#!/bin/bash
set -x
set -euo pipefail

OS_NAME="$(uname -s)"

if [ "${OS_NAME}" = "Darwin" ]; then
	XCODE_DEV="/Users/a58/Downloads/Xcode.app/Contents/Developer"
	SDK="$XCODE_DEV/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
	CLANG="$XCODE_DEV/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"
	CLANGXX="$XCODE_DEV/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++"

	export SDKROOT="$SDK"
	export CC="$CLANG -isysroot $SDK"
	export CXX="$CLANGXX -isysroot $SDK"
	export CFLAGS="-isysroot $SDK"
	export CXXFLAGS="-isysroot $SDK"
	export LDFLAGS="-isysroot $SDK"

	rm -rf build metal_ext.cpython-310-darwin.so
	/Users/a58/Documents/code/nanochat/.venv/bin/python setup.py build_ext --inplace
elif command -v nvcc > /dev/null 2>&1; then
	PYTHON=${PYTHON:-python3}
	CUDA_ARCH=${CUDA_ARCH:-native}   # 例如 RTX 3090 设 sm_86，A100 设 sm_80，RTX 4090 设 sm_89；CUDA 11.6+ 默认 native 自动探测本机 GPU
	PYBIND_INC=$($PYTHON -m pybind11 --includes)
	EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

	rm -rf build cuda_ext*.so
	nvcc -O3 -std=c++17 --use_fast_math -arch="${CUDA_ARCH}" \
	    --shared -Xcompiler "-fPIC -Wall" \
	    ${PYBIND_INC} \
	    cuda_ext.cu -lcudart \
	    -o "cuda_ext${EXT_SUFFIX}"
else
	echo "Unsupported build environment: OS=${OS_NAME} and nvcc not found" >&2
    exit 1
fi
