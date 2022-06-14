'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch


def vertex2face(vertex_features_bxpxk, faces_fx3):

    ##########################################################
    # 1 points
    '''
    pf0_bxfx3 = vertex_features_bxpxk[:, faces_fx3[:, 0], :]
    pf1_bxfx3 = vertex_features_bxpxk[:, faces_fx3[:, 1], :]
    pf2_bxfx3 = vertex_features_bxpxk[:, faces_fx3[:, 2], :]
    face_features_bxfx3k = torch.cat((pf0_bxfx3, pf1_bxfx3, pf2_bxfx3), dim=2)
    '''

    vertex_features_bxf3xk = vertex_features_bxpxk[:, faces_fx3.view(-1)]
    bnum = vertex_features_bxpxk.shape[0]
    knum = vertex_features_bxpxk.shape[2]
    face_features_bxfx3k = vertex_features_bxf3xk.view(bnum, -1, knum * 3)

    return face_features_bxfx3k
