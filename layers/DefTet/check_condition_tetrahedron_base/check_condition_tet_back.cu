/*
'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
*/

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define eps 1e-10
#define SCALE 1.0


template<typename scalar_t>
__host__ __device__ scalar_t dr_cuda_abs(scalar_t a){
	if (a > 0.0){
		return a;
	}
	else{
		return -a;
	}
}

template<typename scalar_t>
__host__ __device__ scalar_t dr_cuda_min_dis_idx(scalar_t a, scalar_t b, scalar_t c){
	scalar_t min_d = a;
	int min_idx = 0;
	if (b < min_d){
		min_d = b;
		min_idx = 1;
	}
	if (c < min_d){
		min_d = c;
		min_idx = 2;
	}
	return min_idx;
}


template<typename scalar_t>
__host__ __device__ scalar_t dr_cuda_mid_distance(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_t* __restrict__ c){
	// calculate the mid distance of a to bc line
	scalar_t a_x = a[0];
	scalar_t a_y = a[1];

	scalar_t b_x = b[0];
	scalar_t b_y = b[1];
	scalar_t c_x = c[0];
	scalar_t c_y = c[1];

	scalar_t mid_x = (b_x + c_x) / 2;
	scalar_t mid_y = (b_y + c_y) / 2;

	scalar_t distance = dr_cuda_abs(a_x - mid_x) + dr_cuda_abs(a_y - mid_y);
	return distance;
}

template<typename scalar_t>
__host__ __device__ scalar_t check_normal(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_t* __restrict__ c){
	// calculate the mid distance of a to bc line
	scalar_t a_x = a[0];
	scalar_t a_y = a[1];

	scalar_t b_x = b[0];
	scalar_t b_y = b[1];
	scalar_t c_x = c[0];
	scalar_t c_y = c[1];

	scalar_t normal = (b_x - a_x) * (c_y - a_y) - (c_x - a_x) * (b_y - a_y);
	return normal;
}

template<typename scalar_t>
__global__ void dr_cuda_backword_kernel_batch(
		const scalar_t* __restrict__ dl_dmindist_bxnxk,
		const scalar_t* __restrict__ grid_bxkx3x2,
		const scalar_t* __restrict__ img_pos_bxnx2,
		scalar_t* __restrict__ gradient_bxnxkx3x2,
		scalar_t* __restrict__ condition_bxnxk, int n_pixel, int n_grid, int bnum, float sigma) {

	// bidx * height + heiidx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

	int pixel_idx = presentthread % n_pixel;
	int bidx = (presentthread - pixel_idx) / n_pixel;

	if (bidx >= bnum || pixel_idx >= n_pixel)
		return;
	
	int total_idx = bidx * n_pixel * n_grid + pixel_idx * n_grid;
	scalar_t pixel_x = img_pos_bxnx2[bidx * n_pixel * 2 + pixel_idx * 2];
	scalar_t pixel_y = img_pos_bxnx2[bidx * n_pixel * 2 + pixel_idx * 2 + 1];
	scalar_t distance_ab = 0.0;
	scalar_t distance_bc = 0.0;
	scalar_t distance_ca = 0.0;
	int min_distance_idx = 0;
	int idx_one = 0;
	int idx_two = 0;
	

	for (int grididx = 0; grididx < n_grid; grididx++){
		// Get the minimum index distance 
		distance_ab = dr_cuda_mid_distance(img_pos_bxnx2 + bidx * n_pixel * 2 + pixel_idx * 2,
													grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2,
													grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 2);
		distance_bc = dr_cuda_mid_distance(img_pos_bxnx2 + bidx * n_pixel * 2 + pixel_idx * 2,
													grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 2,
													grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 4);
		distance_ca = dr_cuda_mid_distance(img_pos_bxnx2 + bidx * n_pixel * 2 + pixel_idx * 2,
													grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 4,
													grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2);

		min_distance_idx = dr_cuda_min_dis_idx(distance_ab, distance_bc, distance_ca);
		
		idx_one = min_distance_idx;
		idx_two = (idx_one + 1) % 3;
		scalar_t dl_dmindist_element = dl_dmindist_bxnxk[bidx * n_pixel * n_grid + pixel_idx * n_grid + grididx];
//		if (check_normal(grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2, grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 2,
//			grid_bxkx3x2 + bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 4) > 0.0){
//				dl_dmindist_element = 0.0; // we don't change the grids with negative normal, the area constraint will change these grids.
//		}

		int mem_grid_idx = bidx * n_grid * 3 * 2 + grididx * 3 * 2;
		int mem_gradient_idx = bidx * n_pixel * n_grid * 3 * 2 + pixel_idx * n_grid * 3 * 2 + grididx * 3 * 2;
		
		scalar_t mid_x = (grid_bxkx3x2[mem_grid_idx + idx_one * 2] +
			grid_bxkx3x2[mem_grid_idx + idx_two * 2]) / 2; 
		
		scalar_t mid_y = (grid_bxkx3x2[mem_grid_idx + idx_one * 2 + 1] +
				grid_bxkx3x2[mem_grid_idx + idx_two * 2 + 1]) / 2; 
		float sign = condition_bxnxk[total_idx + grididx];
        sign = 2.0 * sign - 1.0;
        // Check the direction of the triangle, if it's in the opposite side then we shouldnot back-propogate.
        // Time SCALE for numerical stability.
		scalar_t ax = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2] * SCALE;
		scalar_t ay = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 1] * SCALE;
		scalar_t bx = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 2] * SCALE;
		scalar_t by = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 3] * SCALE;
		scalar_t cx = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 4] * SCALE;
		scalar_t cy = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 5] * SCALE;

		// replace with other variables
		scalar_t m = bx - ax;
		scalar_t p = by - ay;

		scalar_t n = cx - ax;
		scalar_t q = cy - ay;

		scalar_t k3 = m * q - n * p;

		// is this face visible?, if not, we cut gradient for this triangle.

		if (k3 > 0) {
			dl_dmindist_element = 0.0;
		}

		if (pixel_x > mid_x){
			gradient_bxnxkx3x2[mem_gradient_idx + idx_one * 2] = -0.5 * dl_dmindist_element / sigma * sign;
			gradient_bxnxkx3x2[mem_gradient_idx + idx_two * 2] = -0.5 * dl_dmindist_element / sigma * sign;
		}
		else{
			gradient_bxnxkx3x2[mem_gradient_idx + idx_one * 2] = 0.5 * dl_dmindist_element / sigma * sign;
			gradient_bxnxkx3x2[mem_gradient_idx + idx_two * 2] = 0.5 * dl_dmindist_element / sigma * sign;
		}

		if (pixel_y > mid_y){
			gradient_bxnxkx3x2[mem_gradient_idx + idx_one * 2 + 1] = -0.5 * dl_dmindist_element / sigma * sign;
			gradient_bxnxkx3x2[mem_gradient_idx + idx_two * 2 + 1] = -0.5 * dl_dmindist_element / sigma * sign;
		}
		else{
			gradient_bxnxkx3x2[mem_gradient_idx + idx_one * 2 + 1] = 0.5 * dl_dmindist_element / sigma * sign;
			gradient_bxnxkx3x2[mem_gradient_idx + idx_two * 2 + 1] = 0.5 * dl_dmindist_element / sigma * sign;
		}
	}
}

void dr_cuda_backward_batch(at::Tensor dl_dmindist_bxnxk, at::Tensor grid_bxkx3x2, at::Tensor img_pos_bxnx2, at::Tensor gradient_bxnxkx3x2, float sigma,
                             at::Tensor condition_bxnxk) {

	int bnum = grid_bxkx3x2.size(0);
	int n_grid = grid_bxkx3x2.size(1);
	int n_pixel = img_pos_bxnx2.size(1);
	// for fxbxhxw image size
	const int threadnum = 1024;
	const int totalthread = bnum * n_pixel;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	// we exchange block and thread!
	AT_DISPATCH_FLOATING_TYPES(grid_bxkx3x2.type(), "dr_cuda_backward_batch",
			([&] {
				dr_cuda_backword_kernel_batch<scalar_t><<<blocks, threads>>>(
						dl_dmindist_bxnxk.data<scalar_t>(),
						grid_bxkx3x2.data<scalar_t>(),
						img_pos_bxnx2.data<scalar_t>(),
						gradient_bxnxkx3x2.data<scalar_t>(),
						condition_bxnxk.data<scalar_t>(),
					n_pixel, n_grid, bnum, sigma);
			}));

	return;
}

