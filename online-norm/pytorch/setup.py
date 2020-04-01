#!/usr/bin/env python
"""
Released under BSD 3-Clause License,
Copyright (c) 2019 Cerebras Systems Inc.
All rights reserved.
"""
import glob
import os
from os import path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "online_norm_pytorch", "csrc")

    main_source = path.join(extensions_dir, "norm.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (
        torch.cuda.is_available() and CUDA_HOME is not None and os.path.isdir(CUDA_HOME)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "online_norm_pytorch._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="online_norm_pytorch",
    version=0.2,
    author="Vitaliy Chiley",
    author_email="vitaliy@cerebras.net, info@cerebras.net",
    url="https://github.com/Cerebras/online-normalization",
    description="Online Normalization in PyTorch.",
    packages=find_packages(exclude=("tests", "dep_online_norm_pytorch")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
