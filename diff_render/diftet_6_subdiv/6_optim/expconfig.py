'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import configargparse
####
#################################
def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config',
                        is_config_file=False,
                        help='config file path')
    parser.add_argument("--expname",
                        type=str,
                        default='chair',
                        help='experiment name')

    parser.add_argument(
        "--savedir",
        type=str,
        default=
        '/home/wenzheng/largestore/data-generated/nerf/nerf_synthetic/nerf-re',
        help='where to store ckpts and logs')
    parser.add_argument(
        "--datadir",
        type=str,
        default=
        '/home/wenzheng/largestore/data-generated/nerf/nerf_synthetic/nerf_synthetic',
        help='input data directory')
    '''
    parser.add_argument(
        "--savedir",
        type=str,
        default='/u6/a/wenzheng/remote3/dataset-generated/deftet',
        help='where to store ckpts and logs')
    parser.add_argument(
        "--datadir",
        type=str,
        default='/u6/a/wenzheng/remote3/dataset-unzip/nerf_synthetic',
        help='input data directory')
    '''
    #############################################
    # tet options
    parser.add_argument("--tetres",
                        type=int,
                        default=40,
                        help='tet resolution')
    parser.add_argument("--tetcoef",
                        type=float,
                        default=2.5,
                        help='enlarge tet to contain the full scene')
    parser.add_argument("--tetdim",
                        type=int,
                        default=4,
                        help='learnable tet point features')
    parser.add_argument("--tetdim_fixed",
                        type=int,
                        default=0,
                        help='fixed tet point features')

    # train option
    parser.add_argument("--sublevel",
                        type=int,
                        default=2,
                        help='how many subdivision do we do')
    parser.add_argument("--deletenum",
                        type=int,
                        default=1000,
                        help='how many iterations do we carve shape')
    parser.add_argument("--deletethres",
                        type=float,
                        default=1e-3,
                        help='threshold to decide whether we delete the tet')

    parser.add_argument("--optfixnum",
                        type=int,
                        default=3000,
                        help='iterations for trainig deftet with grid fixed')
    parser.add_argument(
        "--lrfix",
        type=int,
        default=5e-2,
        help='learning rate for trainig deftet with grid fixed')
    parser.add_argument("--optmovnum",
                        type=int,
                        default=2000,
                        help='iterations for trainig deftet with grid moved')
    parser.add_argument(
        "--lrmov",
        type=int,
        default=5e-4,
        help='learning rate for trainig deftet with grid moved')

    parser.add_argument(
        "--pixelsampling",
        type=float,
        default=0.04,
        help='learning rate for trainig deftet with grid moved')

    ##################################################
    parser.add_argument("--weights_im_loss",
                        type=float,
                        default=1,
                        help='loss of image, l1, mean, weights')
    parser.add_argument("--weights_mask_loss",
                        type=float,
                        default=2,
                        help='loss of mask, l1, mean, weights')
    parser.add_argument(
        "--weights_mask_reg",
        type=float,
        default=1e-2,
        help='regularizer of point occuapncy, l1, mean, weights')
    parser.add_argument(
        "--weights_occ_lap",
        type=float,
        default=0,
        help='regularizer of point occuapncy, l1, mean, weights')
    parser.add_argument(
        "--weights_color_reg",
        type=float,
        default=0,
        help='regularizer of point color laplacisian, l2, sum, weights')

    parser.add_argument(
        "--weights_point_mov",
        type=float,
        default=1e-2,
        help='regularizer of point movement, l1, mean, weights')
    parser.add_argument(
        "--weights_pointlap",
        type=float,
        default=1e-2,
        help='regularizer of point movement laplacisian, l2, sum, weights')
    parser.add_argument(
        "--weights_tetvariance",
        type=float,
        default=0,
        help='regularizer of tet volume to be uniform, l2, sum, weights')

    ##########################################
    # mesh options
    parser.add_argument(
        "--mesh_only",
        action='store_true',
        help='do not optimize, reload weights and save mesh to a file')
    parser.add_argument(
        "--mesh_grid_size",
        type=int,
        default=80,
        help=
        'number of grid points to sample in each dimension for marching cubes')
    parser.add_argument("--remote",
                        action='store_true',
                        help='add if training remotely')

    # logging/saving options
    parser.add_argument("--i_print",
                        type=int,
                        default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",
                        type=int,
                        default=100,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights",
                        type=int,
                        default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset",
                        type=int,
                        default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",
                        type=int,
                        default=50000,
                        help='frequency of render_poses video saving')

    return parser
