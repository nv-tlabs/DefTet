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
#include <assert.h>
#include <THC/THC.h>
#include <vector>

#define eps 1e-10
#define SCALE 1000.0

//extern THCState * state;

template<typename scalar_t>
 __host__ __device__ __forceinline__ scalar_t dr_cuda_abs(scalar_t a){
	if (a > 0.0){
		return a;
	}
	else{
		return -a;
	}
}

template<typename scalar_t>
 __host__ __device__ __forceinline__ scalar_t dr_cuda_cross_multiple(scalar_t a_x, scalar_t a_y, scalar_t b_x, scalar_t b_y){
	return a_x * b_y - a_y * b_x;
}


template<typename scalar_t>
 __host__ __device__ __forceinline__ scalar_t dr_cuda_divide_non_zero(scalar_t a){

	if (a == 0){
		return eps;
	}
	if (a < 0){
		return a - eps;
	}
	if (a > 0){
		return a + eps;
	}
}


template<typename scalar_t>
__host__ __device__ __forceinline__ scalar_t cuda_dot(scalar_t* a, scalar_t* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename scalar_t>
__host__ __device__ __forceinline__  void cuda_cross(scalar_t* a, scalar_t* b, scalar_t* normal){
    // [a2b3 - a3b2, a3b1-a1b3, a1b2-a2b1]
    normal[0] = a[1] * b[2] - a[2] * b[1];
    normal[1] = a[2] * b[0] - a[0] * b[2];
    normal[2] = a[0] * b[1] - a[1] * b[0];
}


template<typename scalar_t>
__host__ __device__ __forceinline__ void cuda_minus(scalar_t* a, scalar_t* b, scalar_t* result){
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    result[2] = a[2] - b[2];
}

template<typename scalar_t>
__host__ __device__ __forceinline__ void cuda_multiply(scalar_t* a, scalar_t* b, scalar_t* result){

    result[0] = a[0] * b[0];
    result[1] = a[1] * b[1];
    result[2] = a[2] * b[2];
}

template<typename scalar_t>
__host__ __device__ __forceinline__ void cuda_multiply_constant(scalar_t* a, scalar_t b, scalar_t* result){
    result[0] = a[0] * b;
    result[1] = a[1] * b;
    result[2] = a[2] * b;
}

template<typename scalar_t>
__host__ __device__ __forceinline__ void cuda_add(scalar_t* a, scalar_t b, scalar_t* result){
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
    result[2] = a[2] + b[2];
}

template<typename scalar_t>
__host__ __device__ __forceinline__ void cuda_load(scalar_t* a, scalar_t* b){
    a[0] = b[0];
    a[1] = b[1];
    a[2] = b[2];
}

template<typename scalar_t>
__host__ __device__  bool cuda_check_sign(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* d, scalar_t* p){
    scalar_t result_1[3];
    scalar_t result_2[3];
    scalar_t result_3[3];

    cuda_minus(b, a, result_1);
    cuda_minus(c, a, result_2);
    cuda_cross(result_1, result_2, result_3);
    cuda_minus(d, a, result_1);
    scalar_t dotv4 = cuda_dot(result_3, result_1);
    cuda_minus(p, a, result_1);
    scalar_t dotp = cuda_dot(result_3, result_1);
    bool sign_p = dotp > 0;
    bool sign_v = dotv4 > 0;
    return sign_p == sign_v;
}


template<typename scalar_t>
__global__ void dr_cuda_forward_kernel_batch(
		scalar_t* __restrict__ tet_bxfx4x3,
		scalar_t* __restrict__ point_pos_bxnx3,
		scalar_t*  condition_bxnx1,
		scalar_t* __restrict__ bbox_filter_bxfx6,
		int n_batch, int n_point, int n_tet){

	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int point_idx = presentthread % n_point;
	int batch_idx = (presentthread - point_idx) / n_point;

	if (batch_idx >= n_batch || point_idx >= n_point) {
		return;
	}

	/////////////////////////////////////////////////////////////////
	// which point it belongs to
	int base_point_idx = batch_idx * n_point * 3 + point_idx * 3;
	int base_tet_idx = batch_idx * n_tet * 4 * 3;
	int base_tet_idx_bbox = batch_idx * n_tet * 6;

    scalar_t target_point[3];
    scalar_t a[3], b[3], c[3], d[3];
    cuda_load(target_point, point_pos_bxnx3 + base_point_idx);
    scalar_t target_tet = -1.0;


	for (int tet_idx = 0; tet_idx < n_tet; tet_idx++){
		// Check condition of in grid or outside of grid.
//    	scalar_t x_min = bbox_filter_bxfx6[base_tet_idx_bbox + tet_idx * 6 + 0];
//    	scalar_t y_min = bbox_filter_bxfx6[base_tet_idx_bbox + tet_idx * 6 + 1];
//    	scalar_t z_min = bbox_filter_bxfx6[base_tet_idx_bbox + tet_idx * 6 + 2];
//    	scalar_t x_max = bbox_filter_bxfx6[base_tet_idx_bbox + tet_idx * 6 + 3];
//    	scalar_t y_max = bbox_filter_bxfx6[base_tet_idx_bbox + tet_idx * 6 + 4];
//    	scalar_t z_max = bbox_filter_bxfx6[base_tet_idx_bbox + tet_idx * 6 + 5];
//
//    	if (target_point[0] < x_min || target_point[0] > x_max || target_point[1] < y_min || target_point[1] > y_max ||
//    	    target_point[2] < z_min || target_point[2] > z_max){
//    	    continue;
//    	}


        cuda_load(a, tet_bxfx4x3 + base_tet_idx + tet_idx * 4 * 3);
        cuda_load(b, tet_bxfx4x3 + base_tet_idx + tet_idx * 4 * 3 + 3);
        cuda_load(c, tet_bxfx4x3 + base_tet_idx + tet_idx * 4 * 3 + 6);
        cuda_load(d, tet_bxfx4x3 + base_tet_idx + tet_idx * 4 * 3 + 9);

        bool sign_1 = cuda_check_sign(a, b, c, d, target_point);
        bool sign_2 = cuda_check_sign(b, a, d, c, target_point);
        bool sign_3 = cuda_check_sign(c, d, a, b, target_point);
        bool sign_4 = cuda_check_sign(d, c, b, a, target_point);
        if ((sign_1 == sign_2) && (sign_2 == sign_3) && (sign_3 == sign_4)){
            target_tet = (float)(tet_idx);
            break;
        }

//        if ((sign_1 == sign_2) && (sign_2 == sign_3) && (sign_3 == sign_4)){
//
//        }

	}
//	__syncwarp();
//	target_tet = 0.0;
	condition_bxnx1[batch_idx * n_point + point_idx] = target_tet;
}

void dr_cuda_forward_batch(at::Tensor tet_bxfx4x3, at::Tensor point_pos_bxnx3, at::Tensor condition_bxnx1, at::Tensor bbox_filter_bxfx6){

	int n_batch = tet_bxfx4x3.size(0);
	int n_tet = tet_bxfx4x3.size(1);
	int n_point = point_pos_bxnx3.size(1);

	// for fxbxhxw image size
	const int threadnum = 1024;
	const int totalthread = n_batch * n_point;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);


	AT_DISPATCH_FLOATING_TYPES(tet_bxfx4x3.type(), "dr_cuda_forward_batch", ([&] {
		dr_cuda_forward_kernel_batch<scalar_t><<<blocks, threads>>>(
				tet_bxfx4x3.data<scalar_t>(),
				point_pos_bxnx3.data<scalar_t>(),
				condition_bxnx1.data<scalar_t>(),
				bbox_filter_bxfx6.data<scalar_t>(),
				n_batch, n_point, n_tet);
	}));

	return;
}
