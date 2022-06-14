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
from torch.autograd import Function

import sys
sys.path.append('..')
from config import rootdir
sys.path.append('%s/4_render' % rootdir)
from vertex2face import vertex2face

import kaolin as kal
##################################################################
def preprocess_save(tfpoints, tfpfeat_bxpxd):
    featact = nn.Sigmoid()(tfpfeat_bxpxd)
    return featact[:, :1], featact[:, 1:]


def peel2mask(ims_bxpxkxd, imdepth_bxpxkx1=None):

    dnum = ims_bxpxkxd.shape[3]

    immask_bxpxkx1 = ims_bxpxkxd[:, :, :, :1]
    imcolor_bxpxkxc = ims_bxpxkxd[:, :, :, 1:]

    # clip
    eps = 1e-10
    immask_bxpxkx1 = torch.clamp(immask_bxpxkx1, eps, 1.0 - eps)

    xprob_shift = nn.functional.pad(1 - immask_bxpxkx1[:, :, :-1, :],
                                    pad=(0, 0, 1, 0),
                                    mode='constant',
                                    value=1)

    xprob_shiftsum = torch.cumprod(xprob_shift, dim=2)
    xvis = immask_bxpxkx1 * xprob_shiftsum

    xcolor = (imcolor_bxpxkxc * xvis).sum(dim=2)

    if imdepth_bxpxkx1 is not None:
        imdepth_bxpx1 = (imdepth_bxpxkx1 * xvis).sum(dim=2)
    else:
        imdepth_bxpx1 = None

    xvis = xvis.sum(2)

    # white background
    xcolor = xcolor + (1. - xvis)
    if imdepth_bxpx1 is not None:
        imdepth_bxpx1 = imdepth_bxpx1 + -6.0 * (1.0 - xvis)

    return xcolor, xvis, imdepth_bxpx1


def rendermeshcolor(xy_1xpx2,
                    xydep_1xpx2,
                    points3d_bxpx3,
                    points2d_bxpx2,
                    tfpfeat_bxpxd,
                    faces_fx3,
                    viewdir=False,
                    depth=False,
                    istraining=False):

    # unwrap depth & view
    if depth:
        tfdepth_bxpx1 = tfpfeat_bxpxd[:, :, :1]
        tfpfeat_bxpxd = tfpfeat_bxpxd[:, :, 1:]
    assert not viewdir

    # preprocess
    tfpfeat_bxpxd = nn.Sigmoid()(tfpfeat_bxpxd)

    # wrap view and depth
    assert not viewdir
    if depth:
        tfpfeat_bxpxd = torch.cat([tfdepth_bxpx1, tfpfeat_bxpxd], dim=2)

    ######################################################
    # per vertex 2 per face
    points3d_bxfx9 = vertex2face(points3d_bxpx3, faces_fx3)
    points2d_bxfx6 = vertex2face(points2d_bxpx2, faces_fx3)
    tfcolor_bxfx3d = vertex2face(tfpfeat_bxpxd, faces_fx3)

    imcolor_bxpxkx4, _ = kal.render.mesh.deftet_sparse_render(
        xy_1xpx2, xydep_1xpx2, points3d_bxfx9.reshape(points3d_bxfx9.shape[0], points3d_bxfx9.shape[1], 3, 3)[:, :, :, -1],
        points2d_bxfx6.reshape(points2d_bxpx2.shape[0], points2d_bxfx6.shape[1], 3, 2),
        tfcolor_bxfx3d.reshape(tfcolor_bxfx3d.shape[0], tfcolor_bxfx3d.shape[1], 3, -1))

    # unwrap depth & vie1w
    if depth:
        imdepth_bxpxkx1 = imcolor_bxpxkx4[:, :, :, :1]
        imcolor_bxpxkx4 = imcolor_bxpxkx4[:, :, :, 1:]
    else:
        imdepth_bxpxkx1 = None
    assert not viewdir

    imcolor_bxpx3, immask_bxpx1, imdep_bxpx1 = peel2mask(
        imcolor_bxpxkx4, imdepth_bxpxkx1)

    return imcolor_bxpx3, immask_bxpx1, imdep_bxpx1
