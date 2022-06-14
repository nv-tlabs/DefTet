'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import os

for expname in ['chair', 'lego', 'hotdog']:

    for weights_mask_loss in [2, 0]:

        savedir = '/home/wenzheng/largestore/data-generated/nerf/nerf_synthetic/nerf-re'
        datadir = '/home/wenzheng/largestore/data-generated/nerf/nerf_synthetic/nerf_synthetic'

        weights_occ_lap = 0
        weights_color_reg = 0

        if weights_mask_loss == 2:
            weights_mask_reg = 0.01
        elif weights_mask_loss == 0:
            weights_mask_reg = 0.1

        cmd = 'python optim_with_mask_subdiv_from_gridmov.py  \
        --expname %s --savedir %s --datadir %s \
        --weights_mask_loss %.4f --weights_mask_reg %.4f\
        --weights_occ_lap %.4f --weights_color_reg %.4f --i_img 500'\
         % (expname, savedir, datadir, \
            weights_mask_loss, weights_mask_reg, \
            weights_occ_lap, weights_color_reg)
        os.system(cmd)
