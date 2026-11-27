import sys, os, subprocess
from setuptools import setup, Extension
import pybind11
from setuptools.command.build_ext import build_ext

if sys.platform == "darwin":
    class MetalBuildExt(build_ext):
        def build_extensions(self):
            if '.mm' not in self.compiler.src_extensions:
                self.compiler.src_extensions.append('.mm')
            self.compiler.language_map['.mm'] = 'c++'
            super().build_extensions()

    ext_modules = [Extension("metal_ext", ["metal_ext.mm"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-std=c++14", "-Wall", "-O3", "-fobjc-arc"],
        extra_link_args=["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalPerformanceShaders"]
        )]
    cmdclass = {"build_ext": MetalBuildExt}
else:
    class NVCCBuildExt(build_ext):
        def build_extension(self, ext):
            out = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            arch = os.environ.get("CUDA_ARCH", "native")
            subprocess.check_call([
                "nvcc", "-O3", "-std=c++17", "--use_fast_math", f"-arch={arch}",
                "--shared", "-Xcompiler", "-fPIC",
                f"-I{pybind11.get_include()}",
                f"-I{subprocess.check_output(['python3-config', '--includes']).decode().split()[0][2:]}",
                *ext.sources, "-lcudart", "-o", out,
            ])
    ext_modules = [Extension("cuda_ext", ["cuda_ext.cu"])]
    cmdclass = {"build_ext": NVCCBuildExt}

setup(name="autograd_ext", version="0.1", ext_modules=ext_modules, cmdclass=cmdclass)

# class BuildExt(build_ext):
#     def build_extensions(self):
#         if '.mm' not in self.compiler.src_extensions:
#             self.compiler.src_extensions.append('.mm')
#         self.compiler.language_map['.mm'] = 'c++'
#         super().build_extensions()

# ext_modules = [
#     Extension(
#         "metal_ext",
#         ["metal_ext.mm"],
#         include_dirs=[pybind11.get_include()],
#         language="c++",
#         extra_compile_args=[
#             "-O3", 
#             "-Wall", 
#             "-std=c++14", 
#             "-fobjc-arc"
#         ],
#         extra_link_args=[
#             "-framework", "Foundation",
#             "-framework", "Metal",
#             "-framework", "MetalPerformanceShaders",
#         ],
#     ),
# ]

# setup(
#     name="metal_ext",
#     version="0.1",
#     ext_modules=ext_modules,
#     cmdclass={'build_ext': BuildExt}
# )