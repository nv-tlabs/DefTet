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
#include <assert.h>
#include <THC/THC.h>
#include <vector>

#define EPS 1e-15

template<typename scalar_t>
__host__ __device__ scalar_t abs(scalar_t a){
    if (a < 0){
        return -a;
    }
    return a;
}


template<typename scalar_t>
__host__ __device__ bool equal(scalar_t* a, scalar_t* b){

    scalar_t diff = 0.0;

    for (int i = 0; i < 3; i++){
        diff += abs(a[i] - b[i]);
    }
    return diff <= EPS;
}


template<typename scalar_t>
__host__ __device__ bool check_share(scalar_t* face_a, scalar_t* face_b){
    bool find = false;
    scalar_t face_a_a[3] = {0};
    scalar_t face_a_b[3] = {0};
    scalar_t face_b_a[3] = {0};
    scalar_t face_b_b[3] = {0};

	for(int i_a = 0; i_a < 3; i_a++){
	    for(int i = 0; i < 3; i++){
	        face_a_a[i] = face_a[i_a * 3 + i];
	        face_a_b[i] = face_a[((i_a + 1) % 3) * 3 + i];
	    }

	    for (int i_b = 0; i_b < 3; i_b++){

            for(int i = 0; i < 3; i++){
                face_b_a[i] = face_b[i_b * 3 + i];
                face_b_b[i] = face_b[((i_b + 1) % 3) * 3 + i];
	        }


	        if (equal(face_a_a, face_b_a) && equal(face_a_b, face_b_b)){
	            find = true;
	        }
	        if (equal(face_a_a, face_b_b) && equal(face_a_b, face_b_a)){
	            find = true;
	        }
	    }
	}
	return find;
}


template<typename scalar_t>
__global__ void dr_cuda_forward_kernel_batch_min_distance(
        scalar_t* __restrict__ face_fx3x3,
        scalar_t* __restrict__ adj_fxf,
        int n_tet_face,
        int n_max_nei)
		{

	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int face_idx = presentthread;
	if (face_idx >= n_tet_face) {
		return;
	}
    int find_nei = 0;
	scalar_t face[9] = {0};
	for (int i = 0; i< 9; i++){
	    face[i] = face_fx3x3[face_idx * 9 + i];
	}
//	face[0] = face_fx3[face_idx * 3 + 0];
//	face[1] = face_fx3[face_idx * 3 + 1];
//	face[2] = face_fx3[face_idx * 3 + 2];

	scalar_t cmp_face[9];
	for(int i_face = 0; i_face < n_tet_face; i_face++){
	    if (i_face == face_idx){continue;}
	    for (int i = 0; i < 9; i++){
	        cmp_face[i] = face_fx3x3[i_face * 9 + i];
	    }
	    if(check_share(face, cmp_face)){
	        adj_fxf[face_idx * n_max_nei + find_nei] = __int2float_rz(i_face);
	        find_nei += 1;
	    }
	    if (find_nei >= n_max_nei){
	        break;
	    }
	}
}

void dr_cuda_forward_batch(at::Tensor face_fx3x3, at::Tensor adj_fxf){

	int n_tet_face = face_fx3x3.size(0);
	int n_max_nei = adj_fxf.size(1);
	const int threadnum = 512;
	const int totalthread = n_tet_face;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	AT_DISPATCH_FLOATING_TYPES(face_fx3x3.type(), "dr_cuda_forward_kernel_min_distance", ([&] {
		dr_cuda_forward_kernel_batch_min_distance<scalar_t><<<blocks, threads>>>(
		        face_fx3x3.data<scalar_t>(),
		        adj_fxf.data<scalar_t>(),
                n_tet_face,
                n_max_nei);
	}));

	return;
}

