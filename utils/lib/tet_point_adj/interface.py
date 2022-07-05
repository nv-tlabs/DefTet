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
        self.lib = c.cdll.LoadLibrary(os.path.join(ROOT, 'utils/lib/tet_point_adj/run.so'))
        self.lib.run.argtypes = [c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.POINTER(c.c_int32), c.c_int,  c.c_int]

    def run(self, n_point, tet_list, normalize=False):

        assert tet_list.dtype == np.int32

        tet_list = np.ascontiguousarray(tet_list)


        tet_list_p = tet_list.ctypes.data_as(c.POINTER(c.c_int32))
        n_edge = tet_list.shape[0] * 12

        edge = np.zeros((n_edge, 2), dtype=np.int32)
        edge_p = edge.ctypes.data_as(c.POINTER(c.c_int32))
        n_edge = np.zeros(1, dtype=np.int32)
        n_edge_p = n_edge.ctypes.data_as(c.POINTER(c.c_int32))
        self.lib.run(tet_list_p, edge_p, n_edge_p, n_point, tet_list.shape[0])
        v = np.ones(n_edge_p[0])
        # import ipdb
        # ipdb.set_trace()

        new_adj_list = edge[:n_edge[0],:]
        idx = new_adj_list
        v = np.ones(idx.shape[0])
        if normalize:
            adj_m = coo_matrix((v, (idx[:, 0], idx[:, 1])),
                               shape=(n_point, n_point))

            sum_adj = 1.0 / adj_m.sum(axis=-1)
            n_point = sum_adj.shape[0]
            new_idx = list(range(n_point))
            sum_m = coo_matrix((np.asarray(sum_adj).reshape(-1),
                                (new_idx, new_idx)), shape=(n_point, n_point))
            adj = sum_m.dot(adj_m)
            idx = np.asarray(adj.nonzero())
            adj = torch.sparse.FloatTensor(torch.from_numpy(idx).long(
            ), torch.from_numpy(adj.data).float(), torch.Size([n_point, n_point]))
        else:
            idx = torch.from_numpy(idx)
            v = torch.ones(idx.shape[0])
            adj = torch.sparse.FloatTensor(idx.transpose(
                0, 1), v, torch.Size([n_point, n_point]))

        return adj


if __name__ == '__main__':
    f = Tet_point_adj()