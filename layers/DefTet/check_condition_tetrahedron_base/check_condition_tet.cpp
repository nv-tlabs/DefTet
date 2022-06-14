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

// CUDA forward declarations

void dr_cuda_forward_batch(at::Tensor tet_bxfx4x3, at::Tensor point_pos_bxnx3,  at::Tensor condition_bxnx1, at::Tensor bbox_filter_bxfx6);

void dr_forward_batch(at::Tensor tet_bxfx4x3, at::Tensor point_pos_bxnx3,  at::Tensor condition_bxnx1, at::Tensor bbox_filter_bxfx6) {
	CHECK_INPUT(tet_bxfx4x3);
	CHECK_INPUT(point_pos_bxnx3);
	CHECK_INPUT(condition_bxnx1);
	CHECK_INPUT(bbox_filter_bxfx6);
	
	int n_batch = tet_bxfx4x3.size(0);
	int n_tet = tet_bxfx4x3.size(1);
	int n_point = point_pos_bxnx3.size(1);
	CHECK_DIM3(tet_bxfx4x3, n_batch, n_tet, 4, 3);
	CHECK_DIM2(point_pos_bxnx3, n_batch, n_point, 3);
	CHECK_DIM2(condition_bxnx1, n_batch, n_point, 1);
	CHECK_DIM2(bbox_filter_bxfx6, n_batch, n_tet, 6);

	dr_cuda_forward_batch(tet_bxfx4x3, point_pos_bxnx3, condition_bxnx1, bbox_filter_bxfx6);

	return;
}

void dr_cuda_backward_batch(at::Tensor dl_dmindist_bxnxk, at::Tensor grid_bxkx4x2, at::Tensor img_pos_bxnx2, at::Tensor gradient_bxnxkx4x2, float sigma, at::Tensor condition_bxnxk);

void dr_backward_batch(at::Tensor dl_dmindist_bxnxk, at::Tensor grid_bxkx3x2, at::Tensor img_pos_bxnx2, at::Tensor gradient_bxnxkx3x2, float sigma, at::Tensor condition_bxnxk) {

	CHECK_INPUT(dl_dmindist_bxnxk);
	CHECK_INPUT(grid_bxkx3x2);
	CHECK_INPUT(img_pos_bxnx2);
	CHECK_INPUT(gradient_bxnxkx3x2);
	CHECK_INPUT(condition_bxnxk);

	int bnum = grid_bxkx3x2.size(0);
	int n_grid = grid_bxkx3x2.size(1);
	int n_pixel = img_pos_bxnx2.size(1);

	CHECK_DIM3(grid_bxkx3x2, bnum, n_grid, 3, 2);
	CHECK_DIM2(img_pos_bxnx2, bnum, n_pixel, 2);
	CHECK_DIM2(dl_dmindist_bxnxk, bnum, n_pixel, n_grid);
	CHECK_DIM2(condition_bxnxk, bnum, n_pixel, n_grid);
	CHECK_DIM4(gradient_bxnxkx3x2, bnum, n_pixel, n_grid, 3, 2);

	dr_cuda_backward_batch(dl_dmindist_bxnxk, grid_bxkx3x2, img_pos_bxnx2, gradient_bxnxkx3x2, sigma, condition_bxnxk);

	return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &dr_forward_batch, "dr forward batch (CUDA)");
	m.def("backward", &dr_backward_batch, "dr backward batch (CUDA)");
}

