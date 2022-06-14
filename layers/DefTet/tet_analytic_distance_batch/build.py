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
    name='tet_analytic_distance',
    ext_modules=[
        CUDAExtension('tet_analytic_distance', [
            'tet_analytic_distance.cpp',
            'tet_analytic_distance_for.cu',
            'tet_analytic_distance_back.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
