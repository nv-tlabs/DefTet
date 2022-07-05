'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
import os
import ctypes as c
import numpy as np
from scipy.sparse import coo_matrix
import torch

ROOT = os.getcwd()
class Tet_point_adj:
    def __init__(self):
        self.lib = c.cdll.LoadLibrary(os.path.join(ROOT, 'utils/lib/colaps_v/run.so'))
        self.lib.run.argtypes = [c.POINTER(c.c_float), c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.c_int]

    def run(self, point_nx3):
        assert point_nx3.dtype == np.float32

        point = np.ascontiguousarray(point_nx3)
        n_point = point.shape[0]
        map_array = np.zeros(n_point, dtype=np.int32)
        inverse_idx = np.zeros(n_point, dtype=np.int32)

        n_colaps_v = np.zeros(1, dtype=np.int32)
        map_array_p = map_array.ctypes.data_as(c.POINTER(c.c_int32))
        inverse_idx_p = inverse_idx.ctypes.data_as(c.POINTER(c.c_int32))
        n_colaps_v_p = n_colaps_v.ctypes.data_as(c.POINTER(c.c_int32))
        point_p = point.ctypes.data_as(c.POINTER(c.c_float))

        self.lib.run(point_p, map_array_p, inverse_idx_p,  n_colaps_v_p, n_point)
        return map_array, inverse_idx[:n_colaps_v[0]]


if __name__ == '__main__':
    f = Tet_point_adj()