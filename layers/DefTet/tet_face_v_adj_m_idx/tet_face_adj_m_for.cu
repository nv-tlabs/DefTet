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
__host__ __device__ int check_share(scalar_t* point, scalar_t* face){
    bool find = false;
    int find_idx = -1;
    for (int i = 0; i < 3; i++){
        if (equal(point, face + i * 3)){
            find = true;
            find_idx = i;
        }
        if (find){
            break;
        }
    }
    return find_idx;
}

template<typename scalar_t>
__host__ __device__ bool check_exist(scalar_t* point_adj, scalar_t* face_fx3x3,
                                         int find_idx, int find_nei){
    bool find = false;
    for (int i = 0; i < find_nei; i++){
        int point_idx_in_adj = static_cast<int>(point_adj[i]);
        if(equal(face_fx3x3 + find_idx * 3, face_fx3x3 + point_idx_in_adj * 3)){
            find = true;
        }
    }
    return find;
}
//    if(find){
//        return true;
//    }
//    return false;
//    point_adj[find_nei] =  __int2float_rz(find_idx);
//    return find_nei + 1;
//}


template<typename scalar_t>
__global__ void dr_cuda_forward_kernel_batch_min_distance(
        scalar_t* __restrict__ face_fx3x3,
        scalar_t* __restrict__ adj_fxf,
        int n_tet_face,
        int n_max_nei,
        int n_point)
		{
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int point_idx = presentthread;
	if (point_idx >= n_point) {
		return;
	}
    int find_nei = 0;
	scalar_t point[3] = {0};
	for (int i = 0; i< 3; i++){
	    point[i] = face_fx3x3[point_idx * 3 + i];
	}

	scalar_t cmp_face[9];
	bool find_p = false;
	for(int i_face = 0; i_face < n_tet_face; i_face++){
	    for (int i = 0; i < 9; i++){
	        cmp_face[i] = face_fx3x3[i_face * 9 + i];
	    }
	    int shared_idx = check_share(point, cmp_face);

	    if(shared_idx >= 0){
	        find_p = check_exist(adj_fxf + point_idx * n_max_nei, face_fx3x3,
	                                    i_face * 3 + (shared_idx + 1) %3,
	                                    find_nei);
	        if (!find_p){
	            adj_fxf[point_idx * n_max_nei + find_nei] =  __int2float_rz(i_face * 3 + (shared_idx + 1) %3);
	            find_nei += 1;
	        }

	        find_p = check_exist(adj_fxf + point_idx * n_max_nei, face_fx3x3,
	                                    i_face * 3 + (shared_idx + 2) %3,
	                                    find_nei);
	        if (!find_p){
	            adj_fxf[point_idx * n_max_nei + find_nei] =  __int2float_rz(i_face * 3 + (shared_idx + 2) %3);
	            find_nei += 1;
	        }

	    }
	    if (find_nei >= n_max_nei){
	        break;
	    }
	}
}

void dr_cuda_forward_batch(at::Tensor face_fx3x3, at::Tensor adj_fxf){

	int n_tet_face = face_fx3x3.size(0);
	int n_point = n_tet_face * 3;
	int n_max_nei = adj_fxf.size(1);
	const int threadnum = 512;
	const int totalthread = n_point;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	AT_DISPATCH_FLOATING_TYPES(face_fx3x3.type(), "dr_cuda_forward_kernel_min_distance", ([&] {
		dr_cuda_forward_kernel_batch_min_distance<scalar_t><<<blocks, threads>>>(
		        face_fx3x3.data<scalar_t>(),
		        adj_fxf.data<scalar_t>(),
                n_tet_face,
                n_max_nei,
                n_point);
	}));

	return;
}

