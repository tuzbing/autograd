import sys
from pathlib import Path
from setuptools import setup, Extension
import pybind11
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        if '.mm' not in self.compiler.src_extensions:
            self.compiler.src_extensions.append('.mm')
        self.compiler.language_map['.mm'] = 'c++'
        super().build_extensions()


_ROOT = Path(__file__).resolve().parent.parent
ext_modules = [
    Extension(
        "metal_ext",
        ["metal_ext.mm"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-Wall",
            "-std=c++14",
            "-fobjc-arc",
        ],
        extra_link_args=[
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
        ],
    ),
]

setup(
    name="metal_ext",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
