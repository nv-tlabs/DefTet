'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Function

import cv2
import numpy as np
from torch.utils.cpp_extension import load
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

tet_analytic_distance_batch= load(name="tet_analytic_distance_batch",
          sources=[os.path.join(ROOT_DIR, "tet_analytic_distance.cpp"),
                   os.path.join(ROOT_DIR, "tet_analytic_distance_for.cu"),
                   os.path.join(ROOT_DIR, "tet_analytic_distance_back.cu")],
          verbose=True)

import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################
eps = 1e-8
debug = False


# 3
# Inherit from Function
class VarianceFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, gt_point_clouds_bxpx3, face_bxfx3x3, n_face_b):

        # condition is which pixel in which grid
        n_batch = gt_point_clouds_bxpx3.shape[0]
        n_point = gt_point_clouds_bxpx3.shape[1]

        closest_f = torch.zeros(n_batch, n_point, 1, device=gt_point_clouds_bxpx3.device).float()
        closest_d = torch.zeros(n_batch, n_point, 1, device=gt_point_clouds_bxpx3.device).float()
        face_bxfx3x3 = face_bxfx3x3.contiguous()
        gt_point_clouds_bxpx3 = gt_point_clouds_bxpx3.contiguous()
        n_face_b = n_face_b.contiguous()
        tet_analytic_distance_batch.forward(gt_point_clouds_bxpx3, face_bxfx3x3, closest_f, closest_d, n_face_b)

        ctx.save_for_backward(gt_point_clouds_bxpx3, face_bxfx3x3, closest_f)

        return closest_d, closest_f

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, dl_dclosest_d, dl_dcloest_f):
        gt_point_clouds_bxpx3, face_bxfx3x3, closest_f = ctx.saved_tensors
        # condition is which pixel in which grid
        n_batch = gt_point_clouds_bxpx3.shape[0]
        n_point = gt_point_clouds_bxpx3.shape[1]
        n_tet_face = face_bxfx3x3.shape[1]
        dl_dclosest_d = dl_dclosest_d.contiguous()
        # print('Backward')
        dldtet_bxfx3x3 = torch.zeros(n_batch, n_tet_face, 3, 3, device=dl_dclosest_d.device).float()
        tet_analytic_distance_batch.backward(gt_point_clouds_bxpx3, face_bxfx3x3, closest_f, dl_dclosest_d, dldtet_bxfx3x3)
        # import ipdb
        # ipdb.set_trace()
        # print('Finished')
        # tmp = face_bxfx3x3.reshape(-1, 3)
        # print('MIN: ', tmp.min(dim=0)[0])
        # print('MAX: ', tmp.max(dim=0)[0])
        # tmp = dldtet_bxfx3x3.reshape(-1, 3)
        # print('MIN Grad: ', tmp.min(dim=0)[0])
        # print('MAX Grad: ', tmp.max(dim=0)[0])
        return None, dldtet_bxfx3x3, None

###############################################################
tet_analytic_distance_f_batch = VarianceFunc.apply
