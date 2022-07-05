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
class Tet_adj_share:
    def __init__(self):
        self.lib = c.cdll.LoadLibrary(os.path.join(ROOT, 'utils/lib/tet_adj_share/run.so'))
        self.lib.run.argtypes = [c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.c_int,  c.c_int]

    def run(self, tet_list, n_point):

        assert tet_list.dtype == np.int32

        tet_list = np.ascontiguousarray(tet_list)


        tet_list_p = tet_list.ctypes.data_as(c.POINTER(c.c_int32))
        n_face = tet_list.shape[0] * 4
        index_list = np.zeros((n_face * 2, 3), dtype=np.int32)
        index_list_p = index_list.ctypes.data_as(c.POINTER(c.c_int32))
        n_face_edge = np.zeros(1, dtype=np.int32)
        n_face_edge_p = n_face_edge.ctypes.data_as(c.POINTER(c.c_int32))
        self.lib.run(tet_list_p, index_list_p, n_face_edge_p, n_point, tet_list.shape[0])
        v = np.ones(n_face_edge_p[0])

        n_tet = tet_list.shape[0]

        index_list = index_list[:n_face_edge[0] * 2]
        index_value = np.ones(index_list.shape[0])

        adj_list = []
        for i in range(4):
            adj_list.append(coo_matrix((index_value[index_list[:, 2] == i],
                                        (index_list[:, 0][index_list[:, 2] == i],
                                         index_list[:, 1][index_list[:, 2] == i])),
                                       shape=(n_tet, n_tet)))

        return adj_list


if __name__ == '__main__':
    f = Tet_adj_share()
