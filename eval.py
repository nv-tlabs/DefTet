'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from config import OPTIONS
from utils.dataloder_helper import get_collate_fn
from parallel import ParallelWrapper
from torch.utils.data import DataLoader
from utils import mesh_utils
from utils.experiment import Experiment
from utils import tet_utils
import argparse
import utils.dataloder_helper as helpers
import kaolin as kal
from layers.DefTet.deftet import DefTet
from collections import defaultdict
import numpy as np
import os
import torch
import torch.nn as nn
import warnings
from utils.matrix_utils import MySparse
from torchvision import transforms
from layers.pc_model import DeformableTetNetwork
from dataloader import create_dataloader
from utils.point_cloud_utils import iou as point_cloud_iou
from utils.point_cloud_utils import f_score, chamfer_distance, chamfer_distance_l1, hausdorff_distance

warnings.simplefilter("ignore", UserWarning)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHORT_INFO = 'Deformable Grid'
INFO = ''
DEFAULT_FOLDER_PATH = os.path.join(ROOT_DIR, 'experiments')

np.random.seed(1)
torch.random.manual_seed(2)
torch.cuda.manual_seed(3)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to experiment to load')
    parser.add_argument('--threshold', type=float,
                        help='compute F-score', default=0.4)
    parser.add_argument('--step', type=int,
                        help='compute F-score', default=0)

    parser.add_argument('--timing', action='store_true',
                        help='compute F-score', default=False)
    parser.add_argument('--smooth', action='store_true',
                        help='compute F-score', default=False)
    parser.add_argument('--save', action='store_true',
                        help='compute F-score', default=False)
    parser.add_argument('--fix', action='store_true',
                        help='compute F-score', default=False)
    return parser.parse_args()


class Engine(object):
    def __init__(self,
                 cur_epoch=0,
                 timing=None,
                 config=None,
                 dataloader_train=None,
                 dataloader_val=None,
                 mesh_threshold=0.5,
                 smooth=False,
                 train_cat=None,
                 save=False,
                 fix_tet=False):

        self.save = save
        self.mesh_threshold = mesh_threshold
        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.timing = timing
        self.smooth = smooth
        self.train_cat = train_cat
        self.fix_tet = fix_tet
        self.val_loss = []
        self.deftet = DefTet()
        self.cur_epoch = cur_epoch

        vertices_nx3, tetrahedron_fx4, mask = helpers.read_tetrahedron(
            res=self.config.res, root=ROOT_DIR)
        self.init_tet_pos = torch.from_numpy(vertices_nx3).to(
            self.config.device) - 0.5  # to make it centered at zero
        self.init_pos_mask = torch.from_numpy(
            mask).float().to(self.config.device)
        self.init_tet_fx4 = torch.from_numpy(
            tetrahedron_fx4).long().to(self.config.device)

        self.point_adj_sparse = tet_utils.c_tet_to_adj_sparse(
            vertices_nx3, tetrahedron_fx4, normalize=True).to(self.config.device)  # c version is correct :) checked!

        self.point_adj_sparse = MySparse(self.point_adj_sparse)

        tet_face_fx3, tet_facetet_idx_fx2, _, _ = tet_utils.tet_to_face(vertices_nx3.shape[0],
                                                                        tetrahedron_fx4
                                                                        )

        self.tet_face_fx3 = torch.from_numpy(tet_face_fx3).long().cuda()
        self.tet_face_tetidx_fx2 = torch.from_numpy(tet_facetet_idx_fx2).long().cuda()

        self.model = DeformableTetNetwork(
            self.config.device,
            scale_pos=self.config.scale_pos,
            train_def=not (self.config.lambda_def == 0.),
            point_cloud=self.config.point_cloud,
            point_adj_sparse=self.point_adj_sparse,
            use_graph_attention=self.config.use_graph_attention,
            upscale=self.config.upscale,
            use_two_encoder=self.config.use_two_encoder,
            timing=self.config.timing,
            use_lap_layer=self.config.use_lap_layer,
            use_disn=self.config.use_disn,
            scale_pvcnn=self.config.scale_pvcnn,
        )

        inverse_v = nn.Parameter(self.deftet.tet_inverse_v(
            self.init_tet_pos, self.init_tet_fx4))
        inverse_v.requires_grad = False
        self.deftet.inverse_v = inverse_v.cuda()

        self.threshold_list = [0.2, 0.4]

        self.device_count = torch.cuda.device_count()

        self.parallel = ParallelWrapper(
            self.model,
            self.deftet,
            experiment.dir_path('visualization'),
            self.point_adj_sparse,
            self.device_count,
            timing=timing,
            use_two_encoder=self.config.use_two_encoder,
            add_input_noise=self.config.add_input_noise,
            n_point=5000 if self.config.res != 100 else 10000,
            use_lap_layer=self.config.use_lap_layer,
            use_point=self.config.point_cloud)

        assert self.device_count == 1 # evaluate on one GPU


    def load_pretrain(self, pretrain_path, step=None):
        prefix = 'best_'
        post_fix = ''
        if not step is None:
            post_fix = '_' + str(step)
            prefix = ''
        load_path = os.path.join(pretrain_path, prefix + 'decoder_occ'+post_fix+'.pth')
        load_dict = torch.load(load_path)
        self.model.decoder_occ.load_state_dict(load_dict)
        load_path = os.path.join(pretrain_path, prefix + 'decoder_pos'+post_fix+'.pth')
        load_dict = torch.load(load_path)
        if not self.config.baseline:
            self.model.decoder_pos.load_state_dict(load_dict)
        load_path = os.path.join(pretrain_path, prefix + 'encoder'+post_fix+'.pth')
        load_dict = torch.load(load_path)
        self.model.encoder.load_state_dict(load_dict)
        if self.config.use_lap_layer:
            load_path = os.path.join(pretrain_path, 'lap_decoder_pos.pth')
            load_dict = torch.load(load_path)
            self.model.lap_decoder_pos.load_state_dict(load_dict)

    def validate_iou(self):
        self.model.eval()
        with torch.no_grad():
            num_batches = 0
            iou_epoch = dict()
            for cat in self.train_cat:
                iou_epoch[cat] = defaultdict(float)
            iou_epoch['Avg'] = defaultdict(float)

            for i, data in enumerate(self.dataloader_val, 0):
                cat = data['synset'][0]
                imgs = data['imgs'][:, :3].float().to(self.config.device) if 'imgs' in data else None
                points = data['sdf_point'].float().to(self.config.device)
                gt_occ = data['sdf_value'].float().to(self.config.device)
                surface_point = data['sample_points'].float().to(self.config.device)
                cam_rot = data['cam_rot'].float().to(self.config.device) if 'cam_rot' in data else None
                cam_pos = data['cam_pos'].float().to(self.config.device) if 'cam_pos' in data else None
                cam_proj = data['cam_proj'].float().to(self.config.device) if 'cam_proj' in data else None

                all_verts = [v.to(self.config.device).unsqueeze(0).expand(
                    self.device_count, -1, -1) for v in data['verts']]
                all_faces = [v.to(self.config.device).unsqueeze(0).expand(
                    self.device_count, -1, -1) for v in data['faces']]

                init_tet_pos_bxnx3 = self.init_tet_pos.float().unsqueeze(
                    0).expand(surface_point.shape[0], -1, -1)
                init_tet_bxfx4 = self.init_tet_fx4.unsqueeze(
                    0).expand(surface_point.shape[0], -1, -1)
                tet_face_tetidx_bxfx2 = self.tet_face_tetidx_fx2.unsqueeze(
                    0).expand(surface_point.shape[0], -1, -1)
                init_tet_face_bxfx3 = self.tet_face_fx3.unsqueeze(
                    0).expand(surface_point.shape[0], -1, -1)

                # Save some intermidiate results
                amips_energy, edge, area_variance, surface_align, normal_loss, occ_loss, occ_iou, lap, delta_loss, \
                tet_pos, pred_occ_prob, condition, surface, pred_surface, other_chamfer_distance, sum_time = self.parallel(
                        imgs=imgs,
                        init_tet_pos_bxnx3=init_tet_pos_bxnx3,
                        init_tet_bxfx4=init_tet_bxfx4,
                        points=points,
                        surface_point=surface_point,
                        save=False,
                        global_step=i,
                        tet_face_tetidx_bxfx2=tet_face_tetidx_bxfx2,
                        all_verts=all_verts,
                        all_faces=all_faces,
                        return_all=True,
                        inference=True,
                        return_surf = True,
                        tet_face_bxfx3=init_tet_face_bxfx3,
                        cam_pos=cam_pos,
                        cam_rot=cam_rot,
                        cam_proj=cam_proj,
                        pred_threshold=self.mesh_threshold if not self.config.use_lap_layer else self.config.lap_threshold,
                        random_seed=i,
                )
                ####################################################################
                iou_epoch[cat]['surf'] += surface_align.mean().item()
                iou_epoch[cat]['occ_iou'] += occ_iou.mean().item()
                iou_epoch[cat]['lap'] += lap.mean().item()
                iou_epoch[cat]['edge'] += edge.mean().item()
                iou_epoch[cat]['area'] += area_variance.mean().item()
                iou_epoch[cat]['delta'] += delta_loss.mean().item()
                iou_epoch[cat]['amips'] += amips_energy.mean().item()

                mesh_v = tet_pos[0, pred_surface[0].reshape(-1)]
                mesh_f = torch.arange(0, mesh_v.shape[0], device=mesh_v.device, dtype=torch.long).reshape(-1, 3)

                pred_points_occ_prob = kal.ops.mesh.check_sign(mesh_v.unsqueeze(dim=0), mesh_f, points, hash_resolution=512).float()
                gt_occ = (gt_occ > 0.0).float()
                iou_epoch[cat]['iou'] += point_cloud_iou(pred_points_occ_prob, gt_occ, thresh=0.5).item()

                # Get the f-score / chamfer / chamfer l1
                pred_points, _ = kal.ops.mesh.sample_points(
                    mesh_v.unsqueeze(dim=0), mesh_f, 100000)

                # Let's compare two version of codes and check
                tmp_f_score = f_score(surface_point, pred_points, extend=True)
                chamfer_me = chamfer_distance(surface_point, pred_points)
                chamfer_l1 = chamfer_distance_l1(surface_point, pred_points)

                iou_epoch[cat]['chamfer'] += chamfer_me.mean().item()
                iou_epoch[cat]['f_score'] += tmp_f_score.mean().item()
                iou_epoch[cat]['chamfer_me'] += chamfer_me.mean().item()
                iou_epoch[cat]['chamfer_l1'] += chamfer_l1.mean().item()
                mean_hausdorff, max_hausdorff = hausdorff_distance(mesh_v, mesh_f, data['verts'][0].to(self.config.device),
                                                                   data['faces'][0].to(self.config.device),
                                                                   pred_points[0], surface_point[0])
                iou_epoch[cat]['mean_hausdorff'] += mean_hausdorff
                iou_epoch[cat]['max_hausdorff'] += max_hausdorff
                iou_epoch[cat]['num_batches'] += 1

                num_batches += 1

                if num_batches % 50 == 0:
                    print(f'[VAL IoU Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}')
                    iou_epoch['Avg'] = defaultdict(float)
                    for cat in self.train_cat:
                        print(f'{cat}: ', end=' ')
                    print('', end='\n')
                    for t in self.threshold_list:
                        for cat in self.train_cat:
                            out_loss = iou_epoch[cat][t] / float(iou_epoch[cat]['num_batches'] + 1e-10)
                            out_loss *= 100
                            print(f'{out_loss:2.2f}', end=' &')
                            iou_epoch['Avg'][t] += out_loss
                        mean_score = iou_epoch['Avg'][t] / len(self.train_cat)
                        print(f'{mean_score:2.2f}', end=' &')
                        print('', end='\n')

                    for k in ['iou', 'f_score', 'mean_hausdorff', 'max_hausdorff', 'chamfer', 'chamfer_l1', 'chamfer_me']:
                        print(k, end=' ')
                        for cat in self.train_cat:
                            score = iou_epoch[cat][k] / float(iou_epoch[cat]['num_batches'] + 1e-10)
                            score *= 100
                            iou_epoch['Avg'][k] += score
                            print(f'{score:2.3f}', end=' &')

                        mean_score = iou_epoch['Avg'][k] / len(self.train_cat)
                        print(f'{mean_score:2.3f}', end=' &')
                        print('', end='\n')
                ##########

                if not self.timing and not (mesh_v.shape[0] == 0) and self.save:
                    save_name = experiment.dir_path('eval_visualization_all_cat_chair')
                    # print('==> Save for vis')
                    if self.smooth:
                        save_name = experiment.dir_path('eval_visualization_smooth')
                    if self.fix_tet:
                        save_name = experiment.dir_path('eval_visualization_fix')
                    save_name = os.path.join(save_name, data['synset'][0])#####
                    if not os.path.exists(save_name):
                        os.makedirs(save_name)

                    mesh_utils.save_mesh(
                        mesh_v.data.cpu().numpy(), mesh_f.data.cpu().numpy(),
                        save_name + '/pred_occ_%.5f_%s.obj' % (tmp_f_score.mean().item(),
                                                                   data['name'][0].split('/')[-1]))

        print('===> Results')
        with open(experiment.file_path('result_update.txt'), 'a') as f:

            print(f'[VAL IoU Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}')
            iou_epoch['Avg'] = defaultdict(float)


            for cat in self.train_cat:
                print(f'{cat}: ', end=' ')

            for t in self.threshold_list:
                print(t, end=': ')
                f.write(str(t) + ': ')
                for cat in self.train_cat:
                    out_loss = iou_epoch[cat][t] / float(iou_epoch[cat]['num_batches'] + 1e-10)
                    out_loss *= 100
                    print(f'{out_loss:2.2f}', end=' &')
                    f.write(f'{out_loss:2.2f},&')
                    iou_epoch['Avg'][t] += out_loss
                mean_score = iou_epoch['Avg'][t] / len(self.train_cat)
                print(f'{mean_score:2.2f}', end=' &')
                print('', end='\n')
                f.write(f'{mean_score:2.2f} &')
                f.write('\n')

            for k in ['iou', 'f_score', 'mean_hausdorff', 'max_hausdorff', 'chamfer', 'chamfer_l1', 'chamfer_me']:
                print(k, end=': ')
                f.write(k + ': ')
                for cat in self.train_cat:
                    score = iou_epoch[cat][k] / float(iou_epoch[cat]['num_batches'] + 1e-10)
                    score *= 100
                    iou_epoch['Avg'][k] += score
                    print(f'{score:2.2f}', end=' &')
                    f.write(f'{score:2.2f} &')

                mean_score = iou_epoch['Avg'][k] / len(self.train_cat)
                print(f'{mean_score:2.3f}', end=' &')
                f.write(f'{mean_score:2.3f} &')
                print('', end='\n')
                f.write('\n')

def main(experiment, config, state, model_path, mesh_threshold, get_time=False,
         smooth=False, save=False, fix_tet=False, step=0):

    config.c_dim=512

    timing = None
    ####################################### We evaluate one by one #########################
    dataloader_val = create_dataloader(batch_size=1, train=False, only_chairs=False)#####


    train_cat = ['02691156',
                 '02828884',
                 '02933112',
                 '02958343',
                 '03001627',
                 '03211117',
                 '03636649',
                 '03691459',
                 '04090263',
                 '04256520',
                 '04379243',
                 '04401088',
                 '04530566']
    print('==> Init Engine')
    trainer = Engine(timing=timing,
                     config=config,
                     dataloader_val=dataloader_val,
                     dataloader_train=None,
                     mesh_threshold=mesh_threshold,
                     smooth=smooth,
                     train_cat=train_cat,
                     save=save,
                     fix_tet=fix_tet)

    print('==> Load Pretrain')
    if step == 0:
        step = None
    trainer.load_pretrain(model_path, step=step)
    print('Evaluate Scores')
    trainer.validate_iou()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = get_parser()
    experiment = Experiment.load(args.experiment_path, options=OPTIONS)
    experiment.experiment_id = args.experiment_path.split('/')[-1]
    experiment.config.dataset_dir = '/data/shapenet_kaolin'
    experiment.root_path = os.path.join(DEFAULT_FOLDER_PATH, experiment.experiment_id)
    config = experiment.config
    main(experiment, config, experiment.state, args.experiment_path, args.threshold,
         get_time=args.timing, smooth=args.smooth, save=args.save, fix_tet=args.fix, step=args.step)

