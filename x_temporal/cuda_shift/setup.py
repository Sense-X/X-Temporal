import os
import torch
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CppExtension

from setuptools import setup
from setuptools import find_packages


sources = []
headers = []
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/shift_cuda.cpp']
    sources += ['src/cuda/shift_kernel_cuda.cu']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
else:
    raise ValueError('CUDA is not available')

extra_compile_args = {"cxx": []}#'-fopenmp', '-std=c99']
extra_compile_args["nvcc"] = []

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]


ext_module = [CUDAExtension(
    'cudashift',
    sources=sources,
    include_dirs=['src/'],
    define_macros=defines,
    extra_compile_args=extra_compile_args
)]

setup(
    name='CUDASHIFT',
    version='0.2',
    author='Hao Shao',
    ext_modules=ext_module,
    packages=find_packages(exclude=("configs", "tests",)),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension})
