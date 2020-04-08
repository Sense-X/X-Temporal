import os
import torch
from torch.utils.ffi import create_extension


#sources = ['src/shift_cuda.c']
#headers = ['src/shift_cuda.h']
sources = []
headers = []
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/shift_cuda.c']
    headers += ['src/shift_cuda.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/shift_kernel_cuda.cu.o']
    with_cuda = True
else:
    raise ValueError('CUDA is not available')

extra_compile_args = ['-fopenmp', '-std=c99']

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.rtc',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args
)

if __name__ == '__main__':
    ffi.build()
