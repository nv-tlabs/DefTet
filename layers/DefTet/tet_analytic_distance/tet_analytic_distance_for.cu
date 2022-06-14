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
#define SCALE 1.0
#define MAX_DIS 10000.0


template<typename scalar_t>
__host__ __device__ scalar_t cuda_abs(scalar_t a){
	if (a > 0.0){
		return a;
	}
	else{
		return -a;
	}
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_square(scalar_t a){
	return a * a;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_cross_multiple(scalar_t a_x, scalar_t a_y, scalar_t b_x, scalar_t b_y){
	return a_x * b_y - a_y * b_x;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_divide_non_zero(scalar_t a){
	if (a == 0){
		return eps;
	}
	if (a < 0){
		return a - eps;
	}
	if (a > 0){
		return a + eps;
	}
	return eps;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_min_dis(scalar_t a, scalar_t b, scalar_t c, scalar_t d){
	scalar_t min_d = a;
	if (b < min_d){
		min_d = b;
	}
	if (c < min_d){
		min_d = c;
	}
	if (d < min_d){
	    min_d = d;
	}
	return min_d;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_min_dis_three(scalar_t a, scalar_t b, scalar_t c){
	scalar_t min_d = a;
	if (b < min_d){
		min_d = b;
	}
	if (c < min_d){
		min_d = c;
	}
	return min_d;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_dot(scalar_t* a, scalar_t* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename scalar_t>
__host__ __device__ void cuda_cross(scalar_t* a, scalar_t* b, scalar_t* normal){
    // [a2b3 - a3b2, a3b1-a1b3, a1b2-a2b1]
    normal[0] = a[1] * b[2] - a[2] * b[1];
    normal[1] = a[2] * b[0] - a[0] * b[2];
    normal[2] = a[0] * b[1] - a[1] * b[0];
    return;
}


template<typename scalar_t>
__host__ __device__ void cuda_minus(scalar_t* a, scalar_t* b, scalar_t* result){
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    result[2] = a[2] - b[2];
    return;
}

template<typename scalar_t>
__host__ __device__ void cuda_multiply(scalar_t* a, scalar_t* b, scalar_t* result){
    result[0] = a[0] * b[0];
    result[1] = a[1] * b[1];
    result[2] = a[2] * b[2];
    return;
}

template<typename scalar_t>
__host__ __device__  void cuda_multiply_constant(scalar_t* a, scalar_t b, scalar_t* result){
    result[0] = a[0] * b;
    result[1] = a[1] * b;
    result[2] = a[2] * b;
    return ;
}

template<typename scalar_t>
__host__ __device__ void cuda_add(scalar_t* a, scalar_t* b, scalar_t* result){
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
    result[2] = a[2] + b[2];
    return;
}

template<typename scalar_t>
__host__ __device__ void cuda_normalize(scalar_t* a){
    scalar_t length = 0.0;
    length = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    length = cuda_divide_non_zero(length);
    a[0] = a[0] / length;
    a[1] = a[1] / length;
    a[2] = a[2] / length;
    return ;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_distance_point_square(scalar_t* a, scalar_t* b){
    scalar_t length = 0.0;
    length = (a[0] - b[0]) * (a[0] - b[0]) +
             (a[1] - b[1]) * (a[1] - b[1]) +
             (a[2] - b[2]) * (a[2] - b[2]);
    return length;
}

template <typename scalar_t>
__host__ __device__ scalar_t distance_line_square(scalar_t* A, scalar_t* B, scalar_t* P){
    //http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    // the line is A + (B - A) * t
    // t = (P - A)^T(B - A) / (B - A)^T (B - A)
    // d^2 = ||P - A - t * (B - A)||^2
    scalar_t P_A[3] = {0};
    scalar_t B_A[3] = {0};
    cuda_minus(P, A, P_A);
    cuda_minus(B, A, B_A);
    scalar_t t = cuda_dot(P_A, B_A) / cuda_divide_non_zero(cuda_dot(B_A, B_A));
    scalar_t d[3] = {0};
    scalar_t tmp[3] = {0};
    cuda_multiply_constant(B_A, t, tmp);
    cuda_minus(P_A, tmp, d);
    scalar_t distance = cuda_dot(d, d);
    if (t >= 0 && t <= 1){
        return distance; // the closest point is inside of the linesegment
    }

    return -distance; // the closest point is outside of the linesegment

}

template<typename scalar_t>
__host__ __device__ void cuda_line_distance(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p, scalar_t * ret){
    //https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    scalar_t k1 = (b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1]);
    scalar_t k2 = (a[0] - c[0]) * (p[1] - c[1]) + (c[1] - a[1]) * (p[0] - c[0]);
    scalar_t k3 = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);

	if(k3 == 0){
	    ret[0] = -1;
	    return ;
	}

	scalar_t l1 = k1 / k3;
	scalar_t l2 = k2 / k3;
	scalar_t l3 = 1 - l1 - l2;
	scalar_t dis12 = distance_line_square(a, b, p);
	scalar_t dis23 = distance_line_square(b, c, p);
	scalar_t dis13 = distance_line_square(a, c, p);

	if (l1 >= 0 && l2 >= 0 && l3 >= 0){ // lie inside or on the boundary
		scalar_t min_dis_line = cuda_min_dis_three(cuda_abs(dis12), cuda_abs(dis23), cuda_abs(dis13));
		ret[0] = 0;
		ret[1] = min_dis_line;
		return;
	}

    if (dis12 <= 0) dis12 = MAX_DIS;
    if (dis23 <= 0) dis23 = MAX_DIS;
    if (dis13 <= 0) dis13 = MAX_DIS;

	scalar_t min_dis_line = cuda_min_dis_three(dis12, dis23, dis13);

	scalar_t d1 = cuda_distance_point_square(a, p);
	scalar_t d2 = cuda_distance_point_square(b, p);
	scalar_t d3 = cuda_distance_point_square(c, p);

	scalar_t min_dis_point = cuda_min_dis_three(d1, d2, d3);

	if (min_dis_line < min_dis_point){
		ret[0] = 1;
		ret[1] = min_dis_line;
	}
	else{
		ret[0] = 2;
		ret[1] = min_dis_point;
	}
	return ;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_min_triangle_distance(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p){
	// calculate the min distance from a point to triangle in 3d
    // find the intersection point
	// https://math.stackexchange.com/questions/588871/minimum-distance-between-point-and-face
	scalar_t result_1[3] = {0};
    scalar_t result_2[3] = {0};
    scalar_t normal[3] = {0};
    cuda_minus(b, a, result_1);
    cuda_minus(c, a, result_2);
    cuda_cross(result_1, result_2, normal);
    cuda_normalize(normal);

    scalar_t t = cuda_dot(normal, a) - cuda_dot(normal, p);
    scalar_t intersection_p[3] = {0};
    cuda_multiply_constant(normal, t, result_1);
    cuda_add(p, result_1, intersection_p);
//    cuda_minus(p, intersection_p, result_1);
    scalar_t distance_1 = t * t;
    // find the line distance from intersect p to triangle
    scalar_t ret[2] = {0};
    cuda_line_distance(a, b, c, intersection_p, ret);
    scalar_t distance = 0;
    if (ret[0] == 0){
        distance = distance_1;
        return distance;
    }
    if (ret[0] < 0){
        return MAX_DIS;
    }
    distance = distance_1 + ret[1];
	return distance;
}

template<typename scalar_t>
__global__ void dr_cuda_forward_kernel_batch_min_distance(
        scalar_t* __restrict__ gt_point_clouds_bxpx3,
        scalar_t* __restrict__ face_bxfx3x3,
		scalar_t* __restrict__ closest_f,
		scalar_t* __restrict__ closest_d,
		int n_batch, int n_point, int n_tet_face)
		{

	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int point_idx = presentthread % n_point;
	int batch_idx = (presentthread - point_idx) / n_point;
	if (batch_idx >= n_batch || point_idx >= n_point) {
		return;
	}

	/////////////////////////////////////////////////////////////////
	// which point it belongs to
	int point_pos_idx = batch_idx * n_point * 3 + point_idx * 3;
	int tet_base_idx = batch_idx * n_tet_face * 3 * 3;
	scalar_t min_d = 10000.0;
	int min_idx = -1;
	scalar_t point[3];
	point[0] = gt_point_clouds_bxpx3[point_pos_idx];
	point[1] = gt_point_clouds_bxpx3[point_pos_idx + 1];
	point[2] = gt_point_clouds_bxpx3[point_pos_idx + 2];

	scalar_t face[9];
	for(int i_face = 0; i_face < n_tet_face; i_face++){
	    int base_face = batch_idx * n_tet_face * 3 * 3 + i_face * 3 * 3;

        face[0] = face_bxfx3x3[base_face];
        face[0 + 1] = face_bxfx3x3[base_face + 1];
        face[0 + 2] = face_bxfx3x3[base_face + 2];
        face[0 + 3] = face_bxfx3x3[base_face + 3];
        face[0 + 4] = face_bxfx3x3[base_face + 4];
        face[0 + 5] = face_bxfx3x3[base_face + 5];
        face[0 + 6] = face_bxfx3x3[base_face + 6];
        face[0 + 7] = face_bxfx3x3[base_face + 7];
        face[0 + 8] = face_bxfx3x3[base_face + 8];

	    scalar_t dis = cuda_min_triangle_distance(face, face + 3, face + 6, point);
	    if (min_d > dis){
	        min_d = dis;
	        min_idx = i_face;
	    }
	}
	closest_d[batch_idx * n_point + point_idx] = min_d;
	closest_f[batch_idx * n_point + point_idx] = __int2float_rz(min_idx);
}

void dr_cuda_forward_batch(at::Tensor gt_point_clouds_bxpx3, at::Tensor face_bxfx3x3, at::Tensor closest_f, at::Tensor closest_d){

	int n_batch = gt_point_clouds_bxpx3.size(0);
	int n_tet_face = face_bxfx3x3.size(1);
	int n_point = gt_point_clouds_bxpx3.size(1);

	const int threadnum = 512;
	const int totalthread = n_batch * n_point;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	AT_DISPATCH_FLOATING_TYPES(gt_point_clouds_bxpx3.type(), "dr_cuda_forward_kernel_min_distance", ([&] {
		dr_cuda_forward_kernel_batch_min_distance<scalar_t><<<blocks, threads>>>(
		        gt_point_clouds_bxpx3.data<scalar_t>(),
		        face_bxfx3x3.data<scalar_t>(),
				closest_f.data<scalar_t>(),
				closest_d.data<scalar_t>(),
				n_batch, n_point, n_tet_face);
	}));

	return;
}

