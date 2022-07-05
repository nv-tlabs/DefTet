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

ROOT = os.getcwd()
class Tet_face_adj:
    def __init__(self):
        self.lib = c.cdll.LoadLibrary(os.path.join(ROOT, 'utils/lib/tet_face_adj/run.so'))
        self.lib.run.argtypes = [c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.c_int,  c.c_int]

    def run(self, n_point, tet_list):
        assert tet_list.dtype == np.int32

        tet_list = np.ascontiguousarray(tet_list)


        tet_list_p = tet_list.ctypes.data_as(c.POINTER(c.c_int32))
        n_face = tet_list.shape[0] * 4
        face_edge = np.zeros((n_face * 50, 2), dtype=np.int32)
        face_edge_p = face_edge.ctypes.data_as(c.POINTER(c.c_int32))
        n_face_edge = np.zeros(1, dtype=np.int32)
        n_face_edge_p = n_face_edge.ctypes.data_as(c.POINTER(c.c_int32))
        self.lib.run(tet_list_p, face_edge_p, n_face_edge_p, n_point, tet_list.shape[0])
        v = np.ones(n_face_edge_p[0])
        adj = coo_matrix((v, (face_edge[:n_face_edge_p[0], 0], face_edge[:n_face_edge_p[0], 1])),
                         shape=(n_face, n_face)).tocsr()

        return adj


if __name__ == '__main__':
    f = Tet_face_adj()