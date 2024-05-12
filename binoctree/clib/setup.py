import os
from os.path import join
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_clib_root = os.path.dirname(os.path.abspath(__file__))
_clib_srcs = glob(join(_clib_root, "src", "*.cpp")) + glob(join(_clib_root, "src", "*.cu"))

_clib_include_dirs = [
    join(_clib_root, "include")
]

print(_clib_srcs)
print(_clib_include_dirs)

ext_modules = [
    CUDAExtension(
        name='octree_clib',
        sources=_clib_srcs,
        include_dirs=_clib_include_dirs
    )
]
setup(
    name='octree_clib',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)