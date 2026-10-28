set -x

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

rm -rf build metal_dw2_debug.metal
/Users/a58/Documents/code/nanochat/.venv/bin/python setup.py build_ext --inplace
