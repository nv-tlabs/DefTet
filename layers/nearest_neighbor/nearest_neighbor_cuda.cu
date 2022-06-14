/*
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
*/
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__forceinline__ __device__ float square(float a) { return a * a; }

__global__ void kernel(const float *queries, const float *points, int *result,
                       int batch_size, int num_queries, int num_points) {

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = thread_id / num_queries;
  if (batch_id >= batch_size) {
    return;
  }
  // int query_id = thread_id % num_queries;

  // TODO: Improve this min_distance
  float min_distance = 1e20;
  int min_point = 0;
  const float *batch_points = points + (batch_id * num_points) * 3;
  float distance = 0;

  const float *batch_queries = queries + thread_id * 3;
  const float query_x = *batch_queries;
  ++batch_queries;
  const float query_y = *batch_queries;
  ++batch_queries;
  const float query_z = *batch_queries;

  for (int i = 0; i < num_points; ++i) {
    distance = 0;
    distance += square(*batch_points - query_x);
    ++batch_points;
    distance += square(*batch_points - query_y);
    ++batch_points;
    distance += square(*batch_points - query_z);
    ++batch_points;
    if (distance < min_distance) {
      min_point = i;
      min_distance = distance;
    }
  }

  result[thread_id] = min_point;
}

void nearest_neighbor_cuda(const float *queries, const float *points,
                           int *result, int batch_size, int num_queries,
                           int num_points) {

  const int threads_per_block = 512;
  const int total_threads = batch_size * num_queries;
  const int num_blocks = total_threads / threads_per_block + 1;

  const dim3 threads(threads_per_block, 1, 1);
  const dim3 blocks(num_blocks, 1, 1);

  // TODO: There's an upper limit to number of blocks in the grid. In total we
  // can have only at most 33553920 query points, which, if we have 92234 query
  // points per batch, at most 363 batches.

  kernel<<<blocks, threads>>>(queries, points, result, batch_size, num_queries,
                              num_points);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in nearest neighbor CUDA kernel: %s\n",
           cudaGetErrorString(err));
  }
}
