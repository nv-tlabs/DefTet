'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='check_condition_cuda_tet_ori',
    ext_modules=[
        CUDAExtension('check_condition_cuda_tet_ori', [
            'check_condition_tet.cpp',
            'check_condition_tet_for.cu',
            'check_condition_tet_back.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
