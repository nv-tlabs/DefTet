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

#define eps 1e-10
#define SCALE 1.0


#define MAX_DIS  9999999.0

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
__host__ __device__ int cuda_min_dis_idx(scalar_t a, scalar_t b, scalar_t c, scalar_t d){
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
	if (d < min_d){
	    min_d = d;
	    min_idx = 3;
	}
	return min_idx;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_min_dis_ret(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* d){
	scalar_t min_d = a[1];
	scalar_t ret = a[0];
	if (b[1] < min_d){
		min_d = b[1];
		ret = b[0];
	}
	if (c[1] < min_d){
		min_d = c[1];
		ret = c[0];
	}
	if (d[1] < min_d){
	    min_d = d[1];
	    ret = d[0];
	}
	return ret;
}

template<typename scalar_t>
__host__ __device__ scalar_t cuda_min_dis_ret_idx(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* d){
	scalar_t min_d = a[1];
	scalar_t ret = a[2];
	if (b[1] < min_d){
		min_d = b[1];
		ret = b[2];
	}
	if (c[1] < min_d){
		min_d = c[1];
		ret = c[2];
	}
	if (d[1] < min_d){
	    min_d = d[1];
	    ret = d[2];
	}
	return ret;
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
__host__ __device__ scalar_t cuda_min_dis_three_idx(scalar_t a, scalar_t b, scalar_t c){
	scalar_t min_d = a;
	scalar_t min_idx = 0;
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
__host__ __device__ scalar_t cuda_dot(scalar_t* a, scalar_t* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<typename scalar_t>
__host__ __device__ void cuda_cross(scalar_t* a, scalar_t* b, scalar_t* normal){
    // [a2b3 - a3b2, a3b1-a1b3, a1b2-a2b1]
    // http://tutorial.math.lamar.edu/Classes/CalcII/CrossProduct.aspx
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
__host__ __device__ void cuda_gradient_line_distance_with_bary(scalar_t* A, scalar_t* B, scalar_t* P, scalar_t * grad){
	// calculate the min distance from a point to triangle in 3d
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
    scalar_t constant = 2.0;
    scalar_t grad_d[3] = {0};
    cuda_multiply_constant(d, constant, grad_d); // grad_d = d * 2
    cuda_multiply_constant(grad_d, t - 1, grad);
    cuda_multiply_constant(grad_d, -t, grad + 3);

    scalar_t grad_t = -2 * cuda_dot(B_A, P_A) + 2 * t * cuda_dot(B_A, B_A);

    // t = bTa / aTa, a = (B - A); b = (P - A)
    scalar_t grad_PA[3] = {0};
    scalar_t aTa = cuda_dot(B_A, B_A);
    scalar_t bTa = cuda_dot(P_A, B_A);
    cuda_multiply_constant(B_A, grad_t / cuda_divide_non_zero(aTa), grad_PA);

    scalar_t grad_BA[3] = {0};
    cuda_multiply_constant(P_A, grad_t / cuda_divide_non_zero(aTa), tmp);
    scalar_t tmp_2[3] = {0};
    cuda_multiply_constant(B_A, -2 * grad_t * bTa / cuda_divide_non_zero(aTa * aTa), tmp_2);
    cuda_add(tmp, tmp_2, grad_BA);

    cuda_add(grad + 3, grad_BA, grad + 3);
    constant = -1.0;
    cuda_multiply_constant(grad_PA, constant, grad_PA);
    cuda_minus(grad_PA, grad_BA, grad_PA);
    cuda_add(grad, grad_PA, grad);

}

template<typename scalar_t>
__host__ __device__ void cuda_gradient_line_distance(scalar_t* A, scalar_t* B, scalar_t* P, scalar_t * grad){
	// calculate the min distance from a point to triangle in 3d
	//http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    // the line is A + (B - A) * t; t * B + (1 - t) * A
    // t = (P - A)^T(B - A) / (B - A)^T (B - A)
    // d^2 = ||P - A - t * (B - A)||^2
    scalar_t P_A[3] = {0};
    scalar_t B_A[3] = {0};
    cuda_minus(P, A, P_A);
    cuda_minus(B, A, B_A);
    scalar_t t = cuda_dot(P_A, B_A) / cuda_divide_non_zero(cuda_dot(B_A, B_A));
    scalar_t intersection_p[3] = {0};
    scalar_t tmp[3] = {0};
    cuda_multiply_constant(B, t, tmp);
    cuda_multiply_constant(A, 1 - t, intersection_p);
    cuda_add(intersection_p, tmp, intersection_p);

    grad[0] = 2 * (intersection_p[0] - P[0]) * (1 - t);
    grad[1] = 2 * (intersection_p[1] - P[1]) * (1 - t);
    grad[2] = 2 * (intersection_p[2] - P[2]) * (1 - t);

    grad[0] = 2 * (intersection_p[0] - P[0]) * (t);
    grad[1] = 2 * (intersection_p[1] - P[1]) * (t);
    grad[2] = 2 * (intersection_p[2] - P[2]) * (t);

}

template<typename scalar_t>
__host__ __device__ void cuda_gradient_line_distance_sympy(scalar_t* a, scalar_t* b, scalar_t* p, scalar_t * grad){
	// calculate the min distance from a point to triangle in 3d
	//http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    // the line is A + (B - A) * t
    // t = (P - A)^T(B - A) / (B - A)^T (B - A)
    // d^2 = ||P - A - t * (B - A)||^2
    scalar_t ax = a[0];
    scalar_t ay = a[1];
    scalar_t az = a[2];

    scalar_t bx = b[0];
    scalar_t by = b[1];
    scalar_t bz = b[2];

    scalar_t px = p[0];
    scalar_t py = p[1];
    scalar_t pz = p[2];

    grad[0] = (-2*(-2*ax + 2*bx)*(-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-ay + by)*(2*ax - bx - px)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-ay + py - (-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-2*(-2*ax + 2*bx)*(-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-az + bz)*(2*ax - bx - px)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-az + pz - (-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-ax + px - (-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-2*(-2*ax + 2*bx)*(-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-ax + bx)*(2*ax - bx - px)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) + 2*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2);
    grad[1] = (-2*(-ax + bx)*(-2*ay + 2*by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-ax + bx)*(2*ay - by - py)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-ax + px - (-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-2*(-2*ay + 2*by)*(-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-az + bz)*(2*ay - by - py)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-az + pz - (-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-ay + py - (-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-2*(-2*ay + 2*by)*(-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-ay + by)*(2*ay - by - py)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) + 2*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2);
    grad[2] = (-2*(-ax + bx)*(-2*az + 2*bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-ax + bx)*(2*az - bz - pz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-ax + px - (-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-2*(-ay + by)*(-2*az + 2*bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-ay + by)*(2*az - bz - pz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-ay + py - (-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-az + pz - (-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-2*(-2*az + 2*bz)*(-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*(-az + bz)*(2*az - bz - pz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) + 2*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2);
    grad[3] = (-2*(-ax + px)*(-ay + by)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(2*ax - 2*bx)*(-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2))*(-ay + py - (-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-2*(-ax + px)*(-az + bz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(2*ax - 2*bx)*(-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2))*(-az + pz - (-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-ax + px - (-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-2*(-ax + bx)*(-ax + px)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(-ax + bx)*(2*ax - 2*bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)));
    grad[4] = (-2*(-ax + bx)*(-ay + py)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(-ax + bx)*(2*ay - 2*by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2))*(-ax + px - (-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-2*(-ay + py)*(-az + bz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(2*ay - 2*by)*(-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2))*(-az + pz - (-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-ay + py - (-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-2*(-ay + by)*(-ay + py)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(-ay + by)*(2*ay - 2*by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)));
    grad[5] = (-2*(-ax + bx)*(-az + pz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(-ax + bx)*(2*az - 2*bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2))*(-ax + px - (-ax + bx)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-2*(-ay + by)*(-az + pz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(-ay + by)*(2*az - 2*bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2))*(-ay + py - (-ay + by)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2))) + (-az + pz - (-az + bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)))*(-2*(-az + bz)*(-az + pz)/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)) - 2*(-az + bz)*(2*az - 2*bz)*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/pow(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2), 2) - 2*((-ax + bx)*(-ax + px) + (-ay + by)*(-ay + py) + (-az + bz)*(-az + pz))/(pow(-ax + bx, 2) + pow(-ay + by, 2) + pow(-az + bz, 2)));

}


template<typename scalar_t>
__host__ __device__ void cuda_line_distance(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p, scalar_t * ret){
    //https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    scalar_t k1 = (b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1]);
    scalar_t k2 = (a[0] - c[0]) * (p[1] - c[1]) + (c[1] - a[1]) * (p[0] - c[0]);
    scalar_t k3 = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);

	if(k3 == 0){ // not a legal triangle
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
		scalar_t min_dis_line_idx = cuda_min_dis_three_idx(cuda_abs(dis12), cuda_abs(dis23), cuda_abs(dis13));
		ret[0] = 0;
		ret[1] = min_dis_line;
		ret[2] = min_dis_line_idx;
		return;
	}

    if (dis12 <= 0) dis12 = MAX_DIS;
    if (dis23 <= 0) dis23 = MAX_DIS;
    if (dis13 <= 0) dis13 = MAX_DIS;

	scalar_t min_dis_line = cuda_min_dis_three(dis12, dis23, dis13);
	scalar_t min_dis_line_idx = cuda_min_dis_three_idx(dis12, dis23, dis13);

	scalar_t d1 = cuda_distance_point_square(a, p);
	scalar_t d2 = cuda_distance_point_square(b, p);
	scalar_t d3 = cuda_distance_point_square(c, p);

	scalar_t min_dis_point = cuda_min_dis_three(d1, d2, d3);
	scalar_t min_dis_point_idx = cuda_min_dis_three_idx(d1, d2, d3);

	if (min_dis_line < min_dis_point){
		ret[0] = 1;
		ret[1] = min_dis_line;
		ret[2] = min_dis_line_idx;
	}
	else{
		ret[0] = 2;
		ret[1] = min_dis_point;
		ret[2] = min_dis_point_idx;
	}
	return ;
}


template<typename scalar_t>
__host__ __device__ scalar_t cuda_min_triangle_distance(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p, scalar_t* ret, scalar_t* intersection_p){
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
    cuda_multiply_constant(normal, t, result_1);
    cuda_add(p, result_1, intersection_p);
//    cuda_minus(p, intersection_p, result_1);
    scalar_t distance_1 = t * t;
    // find the line distance from intersect p to triangle
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
__host__ __device__ void cuda_gradient_triangle_distance(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p, scalar_t* grad){
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
    scalar_t intersection_p[3] = {0};
    scalar_t t = cuda_dot(normal, a) - cuda_dot(normal, p);
    cuda_multiply_constant(normal, t, result_1);
    cuda_add(p, result_1, intersection_p);

    scalar_t k1 = (b[1] - c[1]) * (intersection_p[0] - c[0]) + (c[0] - b[0]) * (intersection_p[1] - c[1]);
    scalar_t k2 = (a[0] - c[0]) * (intersection_p[1] - c[1]) + (c[1] - a[1]) * (intersection_p[0] - c[0]);
    scalar_t k3 = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);

	if(k3 == 0){ // not a legal triangle
	    return ;
	}


	scalar_t l1 = k1 / k3;
	scalar_t l2 = k2 / k3;
	scalar_t l3 = 1 - l1 - l2;


	grad[0] = 2 * (intersection_p[0] - p[0]) * l1;
    grad[1] = 2 * (intersection_p[1] - p[1]) * l1;
    grad[2] = 2 * (intersection_p[2] - p[2]) * l1;

    grad[3] = 2 * (intersection_p[0] - p[0]) * l2;
    grad[4] = 2 * (intersection_p[1] - p[1]) * l2;
    grad[5] = 2 * (intersection_p[2] - p[2]) * l2;

    grad[6] = 2 * (intersection_p[0] - p[0]) * l3;
    grad[7] = 2 * (intersection_p[1] - p[1]) * l3;
    grad[8] = 2 * (intersection_p[2] - p[2]) * l3;


}

template<typename scalar_t>
__host__ __device__ void cuda_gradient_triangle_distance_with_bary(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p, scalar_t* grad){
	// calculate the min distance from a point to triangle in 3d
    // find the intersection point
	// https://math.stackexchange.com/questions/588871/minimum-distance-between-point-and-face

	scalar_t result_1[3];
    scalar_t result_2[3] = {0};
    scalar_t normal[3] = {0};
    cuda_minus(b, a, result_1);
    cuda_minus(c, a, result_2);
    cuda_cross(result_1, result_2, normal);
    // cuda_normalize(normal); // we don't need to normalize the normal vector and the math still holds.

    scalar_t t = cuda_dot(normal, a) - cuda_dot(normal, p);
    scalar_t intersection_p[3] = {0};
    cuda_multiply_constant(normal, t, result_1);
    cuda_add(p, result_1, intersection_p);

    // dd^2 / dp_0
    scalar_t grad_p0[3] = {0};
    cuda_minus(intersection_p, p, grad_p0);
    scalar_t constant = 2.0;
    cuda_multiply_constant(grad_p0, constant, grad_p0);

    // dp_0 / da, dp_0/db, dp_0/dc
    // dl / da = normal * normal
    scalar_t grad_tmp[3] = {0};
    cuda_multiply_constant(normal, normal[0], grad_tmp);
    grad[0] = cuda_dot(grad_p0, grad_tmp);

    cuda_multiply_constant(normal, normal[1], grad_tmp);
    grad[1] = cuda_dot(grad_p0, grad_tmp);

    cuda_multiply_constant(normal, normal[2], grad_tmp);
    grad[2] = cuda_dot(grad_p0, grad_tmp);

    // dl / d_normal
    scalar_t grad_normal[3] = {0};
    scalar_t tmp = 0;

    tmp = a[0] - p[0];
    cuda_multiply_constant(normal, tmp, grad_tmp);
    grad_tmp[0] = grad_tmp[0] + t;
    grad_normal[0] = cuda_dot(grad_p0, grad_tmp);

    tmp = a[1] - p[1];
    cuda_multiply_constant(normal, tmp, grad_tmp);
    grad_tmp[1] = grad_tmp[1] + t;
    grad_normal[1] = cuda_dot(grad_p0, grad_tmp);

    tmp = a[2] - p[2];
    cuda_multiply_constant(normal, tmp, grad_tmp);
    grad_tmp[2] = grad_tmp[2] + t;
    grad_normal[2] = cuda_dot(grad_p0, grad_tmp);

    // dl / da, dl / db, dl / dc
    // gradient for cross product

    scalar_t grad_cross_a[3] = {0};
    scalar_t grad_cross_b[3] = {0};
    cuda_minus(b, a, result_1);
    cuda_minus(c, a, result_2);
    cuda_cross(result_2, grad_normal, grad_cross_a);
    cuda_cross(result_1, grad_normal, grad_cross_b);

    constant = 1.0;
    cuda_multiply_constant(grad_cross_a, constant, grad + 3);
    cuda_multiply_constant(grad_cross_b, constant, grad + 6);

    cuda_minus(grad, grad_cross_a, grad);
    cuda_minus(grad, grad_cross_b, grad);

	return;
}


template<typename scalar_t>
__host__ __device__ void cuda_gradient_triangle_distance_sympy(scalar_t* a, scalar_t* b, scalar_t* c, scalar_t* p, scalar_t* grad)
{
	scalar_t ax = a[0];
    scalar_t ay = a[1];
    scalar_t az = a[2];

    scalar_t bx = b[0];
    scalar_t by = b[1];
    scalar_t bz = b[2];

    scalar_t cx = c[0];
    scalar_t cy = c[1];
    scalar_t cz = c[2];

    scalar_t px = p[0];
    scalar_t py = p[1];
    scalar_t pz = p[2];
    grad[0] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(2*by - 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*bz + 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(-bz + cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*by - 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*bz + 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(by - cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*by - 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*bz + 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(2*by - 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*bz + 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(-bz + cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*by - 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*bz + 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(by - cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*by - 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*bz + 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)));
    grad[1] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(bz - cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(-2*bx + 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*bz - 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(-2*bx + 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*bz - 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(-bx + cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(-2*bx + 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*bz - 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(bz - cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(-2*bx + 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*bz - 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(-2*bx + 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*bz - 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(-bx + cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(-2*bx + 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*bz - 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)));
    grad[2] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(-by + cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(2*bx - 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*by + 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(bx - cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*bx - 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*by + 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*bx - 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*by + 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(-by + cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(2*bx - 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*by + 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(bx - cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*bx - 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*by + 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*bx - 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*by + 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)));
    grad[3] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(-1.0/2.0*(-2*ay + 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(az - cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(-2*ay + 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(-ay + cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(-2*ay + 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(-1.0/2.0*(-2*ay + 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(az - cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(-2*ay + 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(-ay + cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(-2*ay + 2*cy)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*cz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0));
    grad[4] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(-az + cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ax*(-1.0/2.0*(2*ax - 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*ax - 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(ax - cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*ax - 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(-az + cz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*px*(-1.0/2.0*(2*ax - 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*ax - 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(ax - cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*ax - 2*cx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*cz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0));
    grad[5] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(ay - cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ax*(-1.0/2.0*(-2*ax + 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(2*ay - 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(-ax + cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ay*(-1.0/2.0*(-2*ax + 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(2*ay - 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(-1.0/2.0*(-2*ax + 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(2*ay - 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(ay - cy)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*px*(-1.0/2.0*(-2*ax + 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(2*ay - 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(-ax + cx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*py*(-1.0/2.0*(-2*ax + 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(2*ay - 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(-1.0/2.0*(-2*ax + 2*cx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(2*ay - 2*cy)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0));
    grad[6] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(2*ay - 2*by)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*bz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(-az + bz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*ay - 2*by)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*bz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(ay - by)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*ay - 2*by)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*bz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))*(-1.0/2.0*(2*ay - 2*by)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*bz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(-az + bz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*ay - 2*by)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*bz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(ay - by)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*ay - 2*by)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(-2*az + 2*bz)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0));
    grad[7] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(az - bz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ax*(-1.0/2.0*(-2*ax + 2*bx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*bz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(-1.0/2.0*(-2*ax + 2*bx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*bz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*(-ax + bx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*az*(-1.0/2.0*(-2*ax + 2*bx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*bz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(az - bz)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*px*(-1.0/2.0*(-2*ax + 2*bx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*bz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(-1.0/2.0*(-2*ax + 2*bx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*bz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*(-ax + bx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*pz*(-1.0/2.0*(-2*ax + 2*bx)*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by)) - 1.0/2.0*(2*az - 2*bz)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0));
    grad[8] = (ax*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - px*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)))*(2*ax*(-ay + by)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ax*(-1.0/2.0*(2*ax - 2*bx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*ay + 2*by)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*ay*(ax - bx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) + 2*ay*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*ax - 2*bx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*ay + 2*by)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) + 2*az*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*ax - 2*bx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*ay + 2*by)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*px*(-ay + by)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*px*(-1.0/2.0*(2*ax - 2*bx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*ay + 2*by)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*py*(ax - bx)/sqrt(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2)) - 2*py*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz))*(-1.0/2.0*(2*ax - 2*bx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*ay + 2*by)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0) - 2*pz*((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by))*(-1.0/2.0*(2*ax - 2*bx)*((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz)) - 1.0/2.0*(-2*ay + 2*by)*((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz)))/pow(pow((-ax + bx)*(-ay + cy) + (ax - cx)*(-ay + by), 2) + pow((-ax + bx)*(az - cz) + (-ax + cx)*(-az + bz), 2) + pow((-ay + by)*(-az + cz) + (ay - cy)*(-az + bz), 2), 3.0/2.0));
   }

template<typename scalar_t>
__global__ void dr_cuda_backward_kernel_batch_min_distance(
        scalar_t* __restrict__ gt_point_clouds_bxpx3,
        scalar_t* __restrict__ face_bxfx3x3,
		scalar_t* __restrict__ closest_f,
		scalar_t* __restrict__ dl_dclosest_d,
		scalar_t* __restrict__ dldtet_bxfx3x3,
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
	scalar_t point[3];

	point[0] = gt_point_clouds_bxpx3[point_pos_idx];
	point[1] = gt_point_clouds_bxpx3[point_pos_idx + 1];
	point[2] = gt_point_clouds_bxpx3[point_pos_idx + 2];

    int face_idx = static_cast<int>(closest_f[batch_idx * n_point + point_idx]);

	scalar_t face[9];
    for(int i = 0; i < 9; i++){
        face[i] = face_bxfx3x3[tet_base_idx + face_idx * 9 + i];
    }

    int grad_face_offset = batch_idx * n_tet_face * 3 * 3 + face_idx * 3 * 3;
    scalar_t ret[3] = {0};
    scalar_t intersection_p[3];
    cuda_min_triangle_distance(face, face + 3, face + 6, point, ret, intersection_p);
    scalar_t grad_point = dl_dclosest_d[batch_idx * n_point + point_idx];
    if (ret[0] == 0){
        // Cloest point is on the surface
        // Calculate gradient to the grid.
        scalar_t grad_three_points[9] = {0};

        cuda_gradient_triangle_distance(face ,
                                    face + 3,
                                    face + 6,
                                    point,
                                    grad_three_points);
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 0), (float)(grad_point * grad_three_points[0]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 1), (float)(grad_point * grad_three_points[1]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 2), (float)(grad_point * grad_three_points[2]));

        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 3), (float)(grad_point * grad_three_points[3]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 4), (float)(grad_point * grad_three_points[4]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 5), (float)(grad_point * grad_three_points[5]));

        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 6), (float)(grad_point * grad_three_points[6]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 7), (float)(grad_point * grad_three_points[7]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + 8), (float)(grad_point * grad_three_points[8]));
    }
    if (ret[0] == 1){
        // Cloest point is on the line, perpendicular point is not on the surface
        int min_distance_ret_idx_int = static_cast<int>(ret[2]);
        // minimum distance is to the grid line
        int idx_one = min_distance_ret_idx_int;
        int idx_two = (idx_one + 1) % 3;
        scalar_t grad_line[6] = {0};
        cuda_gradient_line_distance(face + idx_one * 3,
                                    face + idx_two * 3,
                                    point, grad_line);

        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx_one * 3 + 0), (float)(grad_point * grad_line[0]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx_one * 3 + 1), (float)(grad_point * grad_line[1]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx_one * 3 + 2), (float)(grad_point * grad_line[2]));

        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx_two * 3 + 0), (float)(grad_point * grad_line[3]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx_two * 3 + 1), (float)(grad_point * grad_line[4]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx_two * 3 + 2), (float)(grad_point * grad_line[5]));
    }
    if (ret[0] == 2){
        int min_distance_ret_idx_int = static_cast<int>(ret[2]);
        // minimum distance is to the grid line
        int idx = min_distance_ret_idx_int;
        scalar_t grad_local_point[3] = {0};
        cuda_minus(face + idx * 3,
                   point, grad_local_point);
        scalar_t constant = 1.0;
        cuda_multiply_constant(grad_local_point, constant, grad_local_point);

        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx * 3 + 0), (float)(2 * grad_point * grad_local_point[0]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx * 3 + 1), (float)(2 * grad_point * grad_local_point[1]));
        atomicAdd((float *)(dldtet_bxfx3x3 + grad_face_offset + idx * 3 + 2), (float)(2 * grad_point * grad_local_point[2]));

    }
}


void dr_cuda_backward_batch(at::Tensor gt_point_clouds_bxpx3, at::Tensor face_bxfx3x3, at::Tensor closest_f, at::Tensor dl_dclosest_d, at::Tensor dldtet_bxfx3x3) {

	int n_batch = gt_point_clouds_bxpx3.size(0);
	int n_tet_face = face_bxfx3x3.size(1);
	int n_point = gt_point_clouds_bxpx3.size(1);

	const int threadnum = 512;
	const int totalthread = n_batch * n_point;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	// we exchange block and thread!
	AT_DISPATCH_FLOATING_TYPES(gt_point_clouds_bxpx3.type(), "dr_cuda_backward_batch",
			([&] {
				dr_cuda_backward_kernel_batch_min_distance<scalar_t><<<blocks, threads>>>(
				        gt_point_clouds_bxpx3.data<scalar_t>(),
				        face_bxfx3x3.data<scalar_t>(),
						closest_f.data<scalar_t>(),
                        dl_dclosest_d.data<scalar_t>(),
                        dldtet_bxfx3x3.data<scalar_t>(),
                        n_batch, n_point, n_tet_face);
			}));

	return;
}

