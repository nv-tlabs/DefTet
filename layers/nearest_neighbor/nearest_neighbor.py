'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
import os
from torch.utils.cpp_extension import load

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

native = load(name="nearest_neighbor_cuda",
              sources=[os.path.join(ROOT_DIR, "nearest_neighbor.cpp"),
                       os.path.join(ROOT_DIR, "nearest_neighbor_cuda.cu")],
              verbose=True)


class NearestNeighborFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, points):
        batch_size, num_queries, dim = queries.shape
        _, num_points, _ = points.shape
        assert dim == 3, "Currently only 3D points are supported"
        assert batch_size == points.shape[0]
        assert dim == points.shape[2]

        queries = queries.contiguous()
        points = points.contiguous()
        result = torch.zeros(batch_size, num_queries,
                             device=points.device, dtype=torch.int32)
        native.forward(queries, points, result,
                       batch_size, num_queries, num_points, dim)
        result = result.long()

        # ctx.save_for_backward(result)

        return result

    # @staticmethod
    # def backward(ctx, graddist1, graddist2):
    #     ints = ctx.saved_tensors
    #     gradxyz1 = torch.zeros(ints.size())
    #     return gradxyz1

    @staticmethod
    def backward(*args):
        raise NotImplementedError


class NearestNeighbor(torch.nn.Module):
    def forward(self, queries, points):
        """
        queries.shape = (batch_size, num_queries, 3)
        points.shape = (batch_size, num_points, 3)
        return shape = (batch_size, num_queries)
        """
        return NearestNeighborFunction.apply(queries, points)
