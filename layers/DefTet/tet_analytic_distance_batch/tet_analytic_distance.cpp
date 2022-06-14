/*
'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
*/

#include <THC/THC.h>
#include <torch/torch.h>

#include <vector>
#include<stdio.h>
extern THCState *state;
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_DIM3(x, b, h, w, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d), #x " must be same im size")
#define CHECK_DIM2(x, b, f, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == f) && (x.size(2) == d), #x " must be same point size")
#define CHECK_DIM4(x, b, h, w, d, k) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d) && (x.size(4) == k), #x " must be same im size")

void dr_cuda_forward_batch(at::Tensor gt_point_clouds_bxpx3, at::Tensor face_bxfx3x3, at::Tensor closest_f, at::Tensor closest_d,  at::Tensor n_face_b);

void dr_forward_batch(at::Tensor gt_point_clouds_bxpx3, at::Tensor face_bxfx3x3, at::Tensor closest_f, at::Tensor closest_d,  at::Tensor n_face_b) {

	CHECK_INPUT(gt_point_clouds_bxpx3);
	CHECK_INPUT(face_bxfx3x3);
	CHECK_INPUT(closest_f);
	CHECK_INPUT(closest_d);
	CHECK_INPUT(n_face_b);


	int n_batch = gt_point_clouds_bxpx3.size(0);
	int n_point = gt_point_clouds_bxpx3.size(1);
	int n_tet_face = face_bxfx3x3.size(1);

	CHECK_DIM2(gt_point_clouds_bxpx3, n_batch, n_point, 3);
	CHECK_DIM2(closest_d, n_batch, n_point, 1);
	CHECK_DIM2(closest_f, n_batch, n_point, 1);
	CHECK_DIM3(face_bxfx3x3, n_batch, n_tet_face, 3, 3);

	dr_cuda_forward_batch(gt_point_clouds_bxpx3, face_bxfx3x3, closest_f, closest_d, n_face_b);
	return;
}

void dr_cuda_backward_batch(at::Tensor gt_point_clouds_bxpx3, at::Tensor face_bxfx3x3, at::Tensor closest_f, at::Tensor dl_dclosest_d,at::Tensor dldtet_bxfx3x3);

void dr_backward_batch(at::Tensor gt_point_clouds_bxpx3, at::Tensor face_bxfx3x3, at::Tensor closest_f, at::Tensor dl_dclosest_d,at::Tensor dldtet_bxfx3x3) {

	CHECK_INPUT(gt_point_clouds_bxpx3);
	CHECK_INPUT(face_bxfx3x3);
	CHECK_INPUT(closest_f);
	CHECK_INPUT(dl_dclosest_d);
	CHECK_INPUT(dldtet_bxfx3x3);

	int n_batch = gt_point_clouds_bxpx3.size(0);
	int n_tet_face = face_bxfx3x3.size(1);
	int n_point = gt_point_clouds_bxpx3.size(1);

	CHECK_DIM2(gt_point_clouds_bxpx3, n_batch, n_point, 3);
	CHECK_DIM3(face_bxfx3x3, n_batch, n_tet_face, 3, 3);
	CHECK_DIM2(dl_dclosest_d, n_batch, n_point, 1);
	CHECK_DIM2(closest_f, n_batch, n_point, 1);
	CHECK_DIM3(dldtet_bxfx3x3, n_batch, n_tet_face, 3, 3);

	dr_cuda_backward_batch(gt_point_clouds_bxpx3, face_bxfx3x3, closest_f, dl_dclosest_d, dldtet_bxfx3x3);
	return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &dr_forward_batch, "dr forward batch (CUDA)");
	m.def("backward", &dr_backward_batch, "dr backward batch (CUDA)");
}

