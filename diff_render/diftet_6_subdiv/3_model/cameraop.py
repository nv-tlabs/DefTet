'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from __future__ import print_function
from __future__ import division

import torch
import torch.nn
import torch.nn as nn
import torch.autograd


##################################################################
def perspective(points_bxpx3, cameras):
    # perspective, use just one camera
    camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras
    cameratrans_rot_bx3x3 = camera_rot_bx3x3.permute(0, 2, 1)

    # follow pixel2mesh!!!
    # new_p = cam_mat * (old_p - cam_pos)
    points_bxpx3 = points_bxpx3 - camera_pos_bx3.view(-1, 1, 3)
    points_bxpx3 = torch.matmul(points_bxpx3, cameratrans_rot_bx3x3)

    camera_proj_bx1x3 = camera_proj_3x1.view(-1, 1, 3)
    xy_bxpx3 = points_bxpx3 * camera_proj_bx1x3
    xy_bxpx2 = xy_bxpx3[:, :, :2] / xy_bxpx3[:, :, 2:3]

    return points_bxpx3, xy_bxpx2
