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


tet_face_adj_m_idx = load(name="tet_face_adj_m_idx",
          sources=[os.path.join(ROOT_DIR, "tet_face_adj_m.cpp"),
                   os.path.join(ROOT_DIR, "tet_face_adj_m_for.cu"),
                   os.path.join(ROOT_DIR, "tet_face_adj_m_back.cu")],
          verbose=True)

import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################
eps = 1e-8
debug = False


# Inherit from Function
class VarianceFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, face_fx3x3):
        n_face = face_fx3x3.shape[0]
        if n_face == 0:
            return torch.zeros(0, device=face_fx3x3.device)

        n_max_nei = 30
        face_fx3x3 = face_fx3x3.contiguous().float()
        adj_idx = -1 * torch.ones(n_face, n_max_nei, device=face_fx3x3.device).float() # each face can only have maximum 10 neighbor face
        tet_face_adj_m_idx.forward(face_fx3x3, adj_idx)

        # import ipdb
        # ipdb.set_trace()
        idx = torch.arange(0, n_face, device=face_fx3x3.device, dtype=torch.long).int()
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, n_max_nei, 1)

        mask = (adj_idx >= 0)
        adj_idx = adj_idx.int().unsqueeze(-1)

        all_adj_idx = torch.cat([idx, adj_idx], dim=-1)
        all_adj_idx = all_adj_idx[mask]
        all_adj_idx = all_adj_idx.permute(1, 0).long()
        return all_adj_idx


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, dl_dclosest_d):
        return None

###############################################################
tet_face_adj_m_f_idx = VarianceFunc.apply
