'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
from utils.experiment import Option
####3
OPTIONS = {
    'point_cloud': Option(
        type=bool,
        value=False,
        help=''
    ),
    'loader_workers': Option(
        type=int,
        value=16,
        help='Number of workers for the dataloader.'
    ),
    'data_root': Option(
        type=str,
        value='/root/dataset',
    load_value='/root/dataset',
        help=''
    ),
    'shape_train_gt_root': Option(
            type=str,
            value='/root/dataset',
    load_value='/root/dataset',
            help=''
    ),
    'shape_train_ori_gt_root': Option(
            type=str,
            value='/root/dataset',
    load_value='/root/dataset',
            help=''
    ),
    'dataset_dir': Option(
        type=str,
        value='/root/dataset',
        help=''
    ),
    'use_all': Option(
        type=bool,
        value=False,
        help=''
    ),
    'expid': Option(
        type=str,
        value='debug',
        help='Unique experiment identifier.'
    ),
    'pretrain': Option(
        type=str,
        value='',
        help=''
    ),
    'device': Option(
        type=str,
        value='cuda',
        help='Device to use'
    ),
    'categories': Option(
        type=str,
        nargs='+',
        value=['chair'],
        help='list of object classes to use'
    ),

    'epochs': Option(
        type=int,
        value=10000,
        help='Number of train epochs.'
    ),
    'lr': Option(
        type=float,
        value=1e-4,
        help='Learning rate.'
    ),
    'val-every': Option(
        type=int,
        value=5,
        help='Validation frequency (epochs).'
    ),
    'batch_size': Option(
        type=int,
        value=4,
        help='batch size'
    ),
    'print_every': Option(
        type=int,
        value=1000,
        help='Print frequency (batches).'
    ),
    'save_vis_every': Option(
        type=int,
        value=10000,
        help='Save frequency (batches).'
    ),
    'logdir': Option(
        type=str,
        value='log',
        help='Directory to log data to.'
    ),
    'save-model': Option(
        type=bool,
        value=True,
        help='Saves the model and a snapshot of the optimizer state.',
    ),
    'res': Option(
        type=float,
        value=50,
        help='Tetrahedron resolutions'
    ),
    'lambda_surf': Option(
        type=float,
        value=1.0,
        load_value=1.0,
        help='Weight for surface align loss'
    ),
    'lambda_occ': Option(
        type=float,
        value=10,
        help='Weight for occupancy loss'
    ),
    'lambda_def': Option(
        type=float,
        value=1,
        help='Weight for Deformation loss, if set to zero, then no deformation prediction'
    ),
    'lambda_normal': Option(
        type=float,
        value=100,
        help='Weight for Normal loss'
    ),
    'lambda_edge': Option(
        type=float,
        value=0,
        help='Weight for edge length loss'
    ),
    'lambda_delta': Option(
        type=float,
        value=10,
        help='Weight for delta loss'
    ),
    'lambda_amips': Option(
        type=float,
        value=10,
        help='Weight for AMIPS loss'
    ),
    'lambda_lap': Option(
        type=float,
        value=10,
        help='Weight for laplacian loss'
    ),
    'lambda_area': Option(
        type=float,
        value=10000,
        help='Weight for volume loss'
    ),
    'lambda_surf_chamfer': Option(
        type=float,
        value=1,
        help='Weight for surface chamfer loss'
    ),
    'lambda_prob_d': Option(
        type=float,
        value=10.0,
        help='Weight for probability loss (no use for now)'
    ),
    'pow': Option(
        type=float,
        value=4,
        help='Power for volume loss'
    ),
    'detach': Option(
        type=bool,
        value=False,
        help='whether detach the deformation, when predicting the occupancy'
    ),
    'sample_box': Option(
        type=bool,
        value=False,
        help=''
    ),
    'grad_norm': Option(
        type=bool,
        value=False,
        help='whether normalize the gradient'
    ),
    'scale_pos': Option(
        type=bool,
        value=True,
        help='whether scale the pos delta prediction'
    ),
    'save_vis': Option(
        type=bool,
        value=False,
        help='Whether to save visualizations'
    ),
    'sigma': Option(
        type=float,
        value=0.0001,
        help='No use for now'
    ),
    'surface': Option(
        type=bool,
        value=True,
        help=''
    ),
    'timing': Option(
        type=bool,
        value=False,
        help='Whether to perform (and only perform) timing experiment'
    ),
    'voxel_res': Option(
        type=int,
        value=100,
        help='The voxel and surface mesh resolution (IF CHANGED, DATASET MUST REGENERATE)'
    ),
    'z_window_radius': Option(
        type=float,
        value=0.025,
        help='The maximum size of any mesh triangle in the ground truth'
    ),
    'use_surface_prob_loss': Option(
        type=bool,
        value=False,
        help='Whether to use loss utilizing surface probability'
    ),
    'use_old_intersection_test': Option(
        type=bool,
        value=False,
        help='Whether to use old point-mesh intersection test'
    ),
    'use_graph_attention': Option(
        type=bool,
        value=False,
        help='Whether to use graph attention for GCN decoder'
    ),
    'use_surface_dis': Option(
        type=bool,
        value=True,
        help='Whether to use surface distance loss'
    ),
    'optimize_network': Option(
        type=bool,
        value=True,
        help='Whether to optimize network'
    ),
    'upsample': Option(
        type=bool,
        value=False,
        help='Whether to upsample '
    ),
    'upsample_layer': Option(
        type=int,
        value=2,
        help='Number of layers to upsample '
    ),
    'upsample_gt_occ': Option(
        type=bool,
        value=True,
        help='Use GT occ or predict occ to upsample '
    ),
    'use_init_pos_mask': Option(
        type=bool,
        value=False,
        help='Whether to use init_pos_mask'
    ),
'use_point': Option(
        type=bool,
        value=True,
        help='Whether to use positional encoding'
    ),
    'use_pos_encoding': Option(
        type=bool,
        value=True,
        help='Whether to use positional encoding'
    ),
    'use_vert_feat': Option(
        type=bool,
        value=True,
        load_value=False,
        help='Whether to use vertex features for occupancy prediction'
    ),

    'use_init_boundary': Option(
        type=bool,
        value=False,
        help='Whether to use vertex features for occupancy prediction'),

    'alternate_training': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to alternate between training deformation and occupancy'
    ),
    'def_epochs': Option(
        type=int,
        value=500,
        help='Number of epochs for deformation (for alternate_training)'
    ),
    'occ_epochs': Option(
        type=int,
        value=500,
        help='Number of epochs for occupancy (for alternate_training)'
    ),
    'use_learned_def_mask': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use learned deformation mask'
    ),
    'c_dim': Option(
        type=int,
        value=512,
        load_value=256,
        help='Number of epochs for occupancy (for alternate_training)'
    ),
    'use_vertex_loss': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use vertices instead of face points for surface loss (use_surface_dis must be False)'
    ),
    'use_l2_chamfer': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use L2 instead of L1 chamfer loss for surface'
    ),
    'occ_detach_def': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to prevent deformation from getting occupancy gradient signal'
    ),
    'use_init_correspondence': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use initial corrrespondence when calculating the loss (false for now)'
    ),
    'expand_boundary': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to expand the boundary when calculating surface align loss (false for now)'
    ),
    'use_pvcnn_pos_decoder': Option(
        type=bool,
        value=True,
        load_value=False,
        help='Whether to use pvcnn for positional decoder'
    ),
    'use_pvcnn_decoder': Option(
        type=bool,
        value=True,
        load_value=False,
        help='Whether to use pvcnn as a decoder (occ + pos) '
    ),
    'use_gcn_pos_decoder': Option(
        type=bool,
        value=False,
        help='Whether to use new GCN decoder for vertex deformation'
    ),
    'use_pvcnn_occ_decoder': Option(
        type=bool,
        value=True,
        load_value=False,
        help='Whether to use pvcnn as a occ decoder'
    ),
    'use_dvr_pos_decoder': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use new GCN decoder for vertex deformation'
    ),
    'use_dvr_occ_decoder': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use pvcnn as a occ decoder'
    ),
    'baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train occnet baseline'
    ),
    'upscale': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to upscale the feature map when sampling the feature in pvcnn'
    ),
    'use_two_encoder': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use two encoder'
    ),
    'use_apex': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use distributed apex (potentially can speedup, do not work on it for now)'
    ),
    'finetune_occ': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to finetune the occupancy prediction (false for now)'
    ),
    'finetune_pos': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to finetune the occupancy prediction (false for now)'
    ),
    'add_input_noise': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to add input noise for point cloud'
    ),
    'full_scene': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to add input noise for point cloud'
    ),
    'voxel_baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train 3dr2n2 baseline'
    ),
    'voxel_baseline_res': Option(
        type=int,
        value=37,
        load_value=37,
        help='The res for 3dr2n2 baseline'
    ),
    'mesh_baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train pixel2mesh baseline'
    ),
    'meshrcnn_baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train meshrcnn baseline'
    ),
    'disn_baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train meshrcnn baseline'
    ),
    'use_disn': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train meshrcnn baseline'
    ),
'meshrcnn_threshold': Option(
        type=float,
        value=0.9,
        load_value=0.9,
        help='Whether to train meshrcnn baseline'
    ),
    'pretrain_voxel': Option(
        type=str,
        value='',
        load_value='',
        help='Pretrain path for meshrcnn baseline'
    ),
    'occnet_baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train occnet baseline'
    ),
    'dmc_baseline': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train dmc baseline'
    ),
    'use_distributed': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use distributed dataparallel (do not do it for now)'
    ),
    'add_geo_feat': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to add geometric feature for occupancy prediction'
    ),
    'optimize_part': Option(
        type=int,
        value=0,
        load_value=0,
        help='the partition for optimization'
    ),
    'use_img_conv': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use conv for image 3d prediction'
    ),
    'use_dvr_decoder': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use dvr decoder'
    ),
    'use_lap_layer': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to use laplacia n layer'
    ),
    'lap_threshold': Option(
        type=float,
        value=0.4,
        load_value=0.4,
        help='the threshold to get laplacian value'
    ),
    'lambda_lap_v_loss': Option(
        type=float,
        value=10,
        load_value=0,
        help='weight for laplacian loss'
    ),

    'use_projection': Option(
        type=bool,
        value=False,
        load_value=False,
        help='whether to use projection to get feature'
    ),

    'train_car': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train car'
    ),

    'pretrain_occ': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train car'
    ),
    'adaptive_sample': Option(
        type=bool,
        value=True,
        load_value=False,
        help='Whether to train car'
    ),
    'use_occ_encoder': Option(
            type=bool,
            value=False,
            load_value=False,
            help='Whether to train car'
        ),

    'pos_pretrain_path': Option(
        type=str,
        value='',
        load_value='',
        help='Whether to train car'
    ),
    'scale_pvcnn': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train car'
    ),
    'predict_color': Option(
        type=bool,
        value=False,
        load_value=False,
        help='Whether to train car'
    ),
    'resize_input_shape': Option(
            type=bool,
            value=True,
            load_value=True,
            help='Whether to train car'
        ),
    'resize_local_feature_shape': Option(
            type=bool,
            value=True,
            load_value=True,
            help='Whether to train car'
        ),
    'local_rank': Option(
        type=int,
        value=0,
        load_value=0,
        help='Whether to train car'
    ),

}
