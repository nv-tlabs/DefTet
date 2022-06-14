'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import numpy as np
import torch
from torch import nn


def convert_torch_sparse(adj):
    row = adj.row
    col = adj.col
    idx = np.stack([row, col], axis=0)
    shape = adj.shape
    adj = torch.sparse.FloatTensor(torch.from_numpy(idx).long(), torch.from_numpy(adj.data).float(), shape)
    return adj

def sparse_batch_matmul(sparse_matrix, dense_matrix_batch):
    '''
    Args:
        sparse_matrix: (m, n)
        dense_matrix_batch: (b, n, p)
    Returns:
        (b, n, p)
    '''
    b, n, p = dense_matrix_batch.shape
    dense_matrix = dense_matrix_batch.transpose(0, 1).reshape(n, b * p)
    result = torch.sparse.mm(sparse_matrix, dense_matrix)
    return result.reshape(n, b, p).transpose(0, 1)

def cross_dot_torch(a, b):
    normal = torch.zeros_like(a)
    normal[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    normal[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    normal[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return normal

def det_m(m_bx3x3):
    a = m_bx3x3[:, 0, :]
    b = m_bx3x3[:, 1, :]
    c = m_bx3x3[:, 2, :]
    det = torch.sum(a * cross_dot_torch(b, c), dim=-1)
    return det

class MySparse(nn.Module):
    def __init__(self, sparse_m):
        super(MySparse, self).__init__()
        self.indices = sparse_m._indices()
        self.indices.requires_grad = False
        self.values = sparse_m._values()
        self.values.requires_grad = False
        self.shape = sparse_m.size()

    def construct(self):
        return torch.sparse.FloatTensor(self.indices, self.values, self.shape)
