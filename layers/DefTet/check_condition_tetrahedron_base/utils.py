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

check_condition_cuda_tet_base = load(name="check_condition_cuda_tet_base",

          sources=[os.path.join(ROOT_DIR, "check_condition_tet.cpp"),
                   os.path.join(ROOT_DIR, "check_condition_tet_for.cu"),
                   os.path.join(ROOT_DIR, "check_condition_tet_back.cu")],
          verbose=True)

# import check_condition_cuda_tet_ori

import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################
eps = 1e-10
debug = False

############################################3
# Inherit from Function
class TriRender2D(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, tet_bxfx4x3, point_pos_bxnx3):
        n_batch = point_pos_bxnx3.shape[0]
        n_p = point_pos_bxnx3.shape[1]
        tet_bxfx4x3 = tet_bxfx4x3.contiguous()
        # Initialize condition with all negative at first
        condition_bxnx1 = - torch.ones(n_batch, n_p, 1, device=point_pos_bxnx3.device, dtype=torch.float)
        bbox_filter_bxfx6 = torch.cat([torch.min(tet_bxfx4x3, dim=2)[0], torch.max(tet_bxfx4x3, dim=2)[0]], dim=-1)
        check_condition_cuda_tet_base.forward(tet_bxfx4x3,
                                              point_pos_bxnx3,
                                              condition_bxnx1,
                                              bbox_filter_bxfx6)
        return condition_bxnx1

    # This function has only a single output, so it gets only one gradient
    @staticmethod 
    def backward(ctx, condition_bxnx1):

        return None, None


###############################################################
check_condition_f_base = TriRender2D.apply

