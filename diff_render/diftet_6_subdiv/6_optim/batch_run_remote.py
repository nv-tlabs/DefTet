'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import os

for weights_occ_lap in [1e-4, 1e-3]:
    for weights_color_lap in [1e-5, 1e-4]:
        for expname in ['chair', 'lego', 'hotdog']:

            savedir = '/u6/a/wenzheng/remote3/dataset-generated/deftet-2'
            datadir = '/u6/a/wenzheng/remote3/dataset-unzip/nerf_synthetic'
            # savedir = '/home/wenzheng/largestore/data-generated/nerf/nerf_synthetic/nerf-re'
            # datadir = '/home/wenzheng/largestore/data-generated/nerf/nerf_synthetic/nerf_synthetic'

            # weights_occ_lap = 0

            cmd = 'python optim_with_mask_subdiv_from_gridmov_paramstuend.py  \
            --expname %s --savedir %s --datadir %s \
            --weights_occ_lap %.4f --weights_color_reg %.4f --i_img 500 \
            --remote' \
             % (expname, savedir, datadir, \
                weights_occ_lap, weights_color_lap)
            os.system(cmd)
