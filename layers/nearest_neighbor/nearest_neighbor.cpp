/*
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
*/
#include <THC/THC.h>
#include <torch/torch.h>

#include <iostream>
#include <stdio.h>
#include <vector>

extern THCState *state;
using namespace std;

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(x, b, f, d)                                                  \
  AT_ASSERTM((x.size(0) == b) && (x.size(1) == f) && (x.size(2) == d),         \
             #x " must be same point size")

void nearest_neighbor_cuda(const float *queries, const float *points,
                           int *result, int batch_size, int num_queries,
                           int num_points);

void nearest_neighbor(const at::Tensor queries, const at::Tensor points,
                      at::Tensor result, int batch_size, int num_queries,
                      int num_points, int dim) {
  AT_ASSERTM(dim == 3, "Currently only 3D points are supported");

  CHECK_INPUT(queries);
  CHECK_INPUT(points);
  CHECK_INPUT(result);

  // NOTE: Dimension checking are done on the Python side.

  nearest_neighbor_cuda(queries.data<float>(), points.data<float>(),
                        result.data<int>(), batch_size, num_queries,
                        num_points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &nearest_neighbor, "Compute nearest neighbor indices");
}
