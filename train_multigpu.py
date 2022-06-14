'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
from config import OPTIONS
from collections import defaultdict
from layers.DefTet.deftet import DefTet
from parallel import ParallelWrapper
from tensorboardX import SummaryWriter
from utils.experiment import Experiment

from utils import tet_utils
import utils.dataloder_helper as helpers
import json
import kaolin as kal
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.matrix_utils import MySparse
from layers.pc_model import DeformableTetNetwork
from dataloader import create_dataloader
from datetime import datetime
from utils.point_cloud_utils import iou as point_cloud_iou
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SHORT_INFO = 'Deformable Grid'
INFO = ''
DEFAULT_FOLDER_PATH = os.path.join(ROOT_DIR, 'experiments')


class Engine(object):
    def __init__(self,
                 cur_epoch=0,
                 timing=None,
                 config=None,
                 dataloader_train=None,
                 dataloader_val=None,
                 experiment=None):

        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.cur_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 0
        self.global_step = 0
        self.experiment = experiment

        self.writer = SummaryWriter(
                self.experiment.dir_path(
                    os.path.join('losses', 'log')))

        deftet = DefTet()

        self.timing = timing

        vertices_nx3, tetrahedron_fx4, mask = helpers.read_tetrahedron(
            res=self.config.res, root=ROOT_DIR)
        self.init_tet_pos = torch.from_numpy(vertices_nx3).to(
            self.config.device) - 0.5  # to make it centered at zero
        self.init_pos_mask = torch.from_numpy(
            mask).float().to(self.config.device)
        self.init_tet_fx4 = torch.from_numpy(
            tetrahedron_fx4).long().to(self.config.device)
        self.deftet = deftet
        self.point_adj_sparse = tet_utils.c_tet_to_adj_sparse(
            vertices_nx3, tetrahedron_fx4, normalize=True).to(self.config.device)  # c version is correct :) checked!

        self.point_adj_sparse = MySparse(self.point_adj_sparse)

        tet_face_fx3, tet_facetet_idx_fx2, _, _ = tet_utils.tet_to_face(vertices_nx3.shape[0],
                                                              tetrahedron_fx4
                                                              )

        self.tet_face_fx3 = torch.from_numpy(tet_face_fx3).long().cuda()
        self.tet_face_tetidx_fx2 = torch.from_numpy(tet_facetet_idx_fx2).long().cuda()

        print('Vertices Shape: ', self.init_tet_pos.shape)
        print('Tet shape: ', self.init_tet_fx4.shape)
        print('Face shape: ', self.tet_face_fx3.shape)

        self.model = DeformableTetNetwork(
            self.config.device,
            scale_pos=self.config.scale_pos,
            train_def=not (self.config.lambda_def == 0.),
            point_cloud=self.config.point_cloud,
            point_adj_sparse=self.point_adj_sparse,
            use_graph_attention=self.config.use_graph_attention,
            upscale=self.config.upscale,
            use_two_encoder= self.config.use_two_encoder,
            timing=self.config.timing,
            use_lap_layer=self.config.use_lap_layer,
            use_disn=self.config.use_disn,
            scale_pvcnn=self.config.scale_pvcnn,
        )

        self.threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]

        inverse_v = nn.Parameter(self.deftet.tet_inverse_v(
            self.init_tet_pos, self.init_tet_fx4))
        inverse_v.requires_grad = False
        if len(self.config.pretrain) > 1:
            self.load_pretrain()
        self.deftet.inverse_v = inverse_v.cuda()
        self.device_count = torch.cuda.device_count()

        self.parameters = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.parameters.append(p)
                print(name)

        self.optimizer = optim.Adam(self.parameters, lr=self.config.lr)

        self.experiment.dir_path('visualization')

        self.parallel = ParallelWrapper(
            self.model,
            self.deftet,
            self.experiment.dir_path('visualization'),
            self.point_adj_sparse,
            self.device_count,
            timing=timing,
            use_two_encoder=self.config.use_two_encoder,
            add_input_noise=self.config.add_input_noise,
            n_point=5000 if self.config.res != 100 else 10000,
            use_lap_layer=self.config.use_lap_layer,
            use_point=self.config.point_cloud)

        if self.device_count > 1:
            self.device_ids = [i for i in range(self.device_count)]
            self.parallel = nn.DataParallel(
                self.parallel, device_ids=self.device_ids)
            print('Using mutiple GPUs: ', self.device_ids)

    def weight_clip(self):
        torch.nn.utils.clip_grad_norm_(self.parameters, 40)

    def load_pretrain(self):
        load_path = os.path.join(self.config.pretrain, 'best_decoder_occ.pth')
        load_dict = torch.load(load_path)
        self.model.decoder_occ.load_state_dict(load_dict)

        load_path = os.path.join(self.config.pretrain, 'best_decoder_pos.pth')
        load_dict = torch.load(load_path)
        self.model.decoder_pos.load_state_dict(load_dict)

        load_path = os.path.join(self.config.pretrain, 'best_encoder.pth')
        load_dict = torch.load(load_path)
        self.model.encoder.load_state_dict(load_dict)

    def get_optim(self):
        return self.optimizer

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        self.model.train()

        for i, data in enumerate(self.dataloader_train, 0):
            self.get_optim().zero_grad()

            imgs = data['imgs'][:, :3].float().to(self.config.device) if 'imgs' in data else None
            points = data['sdf_point'].float().to(self.config.device) if \
                'sdf_point' in data else None
            surface_point = data['sample_points'].float().to(self.config.device)

            all_verts = [v.to(self.config.device).unsqueeze(0).expand(
                self.device_count, -1, -1) for v in data['verts']]
            all_faces = [v.to(self.config.device).unsqueeze(0).expand(
                self.device_count, -1, -1) for v in data['faces']]

            cam_rot = data['cam_rot'].float().to(self.config.device) if 'cam_rot' in data else None
            cam_pos = data['cam_pos'].float().to(self.config.device) if 'cam_pos' in data else None
            cam_proj = data['cam_proj'].float().to(self.config.device) if 'cam_proj' in data else None

            save = self.global_step % self.config.save_vis_every == 0 and self.config.save_vis

            init_tet_pos_bxnx3 = self.init_tet_pos.float().unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)
            init_tet_pos_mask = self.init_pos_mask.float().unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)
            init_tet_bxfx4 = self.init_tet_fx4.unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)
            tet_face_tetidx_bxfx2 = self.tet_face_tetidx_fx2.unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)
            init_tet_face_bxfx3 = self.tet_face_fx3.unsqueeze(
                0).expand(surface_point.shape[0], -1, -1)

            if not self.config.use_init_pos_mask:
                init_tet_pos_mask = None

            if not save:
                amips_energy, edge, area_variance, surface_align, normal_loss, \
                occ_loss, lap, delta_loss, other_chamfer_distance, lap_v_loss = self.parallel(
                    imgs=imgs,
                    init_tet_pos_bxnx3=init_tet_pos_bxnx3,
                    init_tet_bxfx4=init_tet_bxfx4,
                    points=points,
                    surface_point=surface_point,
                    save=save,
                    global_step=self.global_step,
                    tet_face_tetidx_bxfx2=tet_face_tetidx_bxfx2,
                    all_verts=all_verts,
                    all_faces=all_faces,
                    return_all=False,
                    tet_face_bxfx3=init_tet_face_bxfx3,
                    init_pos_mask=init_tet_pos_mask,
                    cam_pos=cam_pos,
                cam_rot=cam_rot,
                cam_proj=cam_proj,
                pred_threshold=self.config.lap_threshold)

            else:
                amips_energy, edge, area_variance,  surface_align, normal_loss, \
                occ_loss, lap, delta_loss, tet_pos, z, encoding_occ, pred_points_occ_prob, gt_occ, other_chamfer_distance, \
                latent, lap_v_loss = self.parallel(
                    imgs=imgs,
                    init_tet_pos_bxnx3=init_tet_pos_bxnx3,
                    init_tet_bxfx4=init_tet_bxfx4,
                    points=points,
                    surface_point=surface_point,
                    save=save ,
                    global_step=self.global_step,
                    tet_face_tetidx_bxfx2=tet_face_tetidx_bxfx2,
                    all_verts=all_verts,
                    all_faces=all_faces,
                    return_all=True,
                    tet_face_bxfx3=init_tet_face_bxfx3,
                    init_pos_mask=init_tet_pos_mask,
                    cam_pos=cam_pos,
                    cam_rot=cam_rot,
                    cam_proj=cam_proj,
                    pred_threshold=self.config.lap_threshold
                )

            surface_align = surface_align.mean()
            area_variance = area_variance.mean()
            normal_loss = normal_loss.mean()
            edge = edge.mean()
            amips = amips_energy.mean()
            other_chamfer_distance = other_chamfer_distance.mean()
            lap_v_loss = lap_v_loss.mean()

            lap = lap.mean()
            delta_loss = delta_loss.mean()
            occ_loss = occ_loss.mean()

            deform_loss = area_variance * self.config.lambda_area + \
                          edge * self.config.lambda_edge + \
                          lap * self.config.lambda_lap + \
                          surface_align * self.config.lambda_surf + \
                          delta_loss * self.config.lambda_delta + \
                          normal_loss * self.config.lambda_normal + \
                          amips * self.config.lambda_amips + \
                          other_chamfer_distance * self.config.lambda_surf_chamfer +\
                            lap_v_loss * self.config.lambda_lap_v_loss

            loss = 0
            if self.config.lambda_occ > 0.0:
                loss += occ_loss * self.config.lambda_occ

            if self.config.lambda_def > 0.0 and (not self.config.finetune_occ):
                loss += deform_loss * self.config.lambda_def


            loss.backward()

            if self.config.grad_norm:
                self.weight_clip()
            self.get_optim().step()

            loss_epoch += float(loss.item())
            num_batches += 1
            if (self.global_step % 10 == 0):
                self.writer.add_scalar(
                    'volumn', area_variance.item(), self.global_step)
                self.writer.add_scalar(
                    'edge_lenth', edge.item(), self.global_step)
                self.writer.add_scalar('lap', lap.item(), self.global_step)
                self.writer.add_scalar(
                    'surf', surface_align.item(), self.global_step)
                self.writer.add_scalar(
                    'delta_loss', delta_loss.item(), self.global_step)
                self.writer.add_scalar(
                    'occ_loss', occ_loss.item(), self.global_step)
                self.writer.add_scalar(
                    'normal', normal_loss.item(), self.global_step)
                self.writer.add_scalar(
                    'amips', amips.item(), self.global_step)
                self.writer.add_scalar(
                    'surf_chamfer', other_chamfer_distance.mean().item(), self.global_step)
                self.writer.add_scalar(
                    'lap_v_loss', lap_v_loss.mean().item(), self.global_step)

            if (self.global_step % self.config.print_every == 0):
                with torch.no_grad():
                    message = '[%s] [TRAIN] Epoch: %d, Batch: %d, Deform_loss: %.5f, Occ_loss: %.5f' % (
                        datetime.now(), self.cur_epoch, i, deform_loss.item(), occ_loss.item())
                    message += ' Volume: %.20f, Edge: %.10f, Lap: %.5f, Delta: %.5f, Surf: %.5f, Surf Chamfer: %.5f' % (
                        area_variance.item(), edge.item(), lap.item(), delta_loss.item(), surface_align.item(),
                        other_chamfer_distance.mean().item())
                    message += ' Normal: %.5f' % (normal_loss.item())
                    message += ' AMIPS: %.5f' % (amips.item())
                    message += ' Lap v: %.5f' % (
                        float(lap_v_loss.mean().item()))
                    print(message)

            self.global_step += 1

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate_iou(self):
        self.model.eval()

        with torch.no_grad():
            num_batches = 0
            iou_epoch = defaultdict(float)

            for i, data in tqdm(enumerate(self.dataloader_val, 0)):
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

                amips_energy, edge, area_variance, surface_align, normal_loss, \
                    occ_loss, occ_iou, lap, delta_loss, tet_pos, pred_occ_prob, condition, other_chamfer_distance= self.parallel(
                        imgs=imgs,
                        init_tet_pos_bxnx3=init_tet_pos_bxnx3,
                        init_tet_bxfx4=init_tet_bxfx4,
                        points=points,
                        surface_point=surface_point,
                        save=False,
                        global_step=self.global_step,
                        tet_face_tetidx_bxfx2=tet_face_tetidx_bxfx2,
                        all_verts=all_verts,
                        all_faces=all_faces,
                        return_all=True,
                        inference=True,
                        tet_face_bxfx3=init_tet_face_bxfx3,
                        cam_pos=cam_pos,
                        cam_rot=cam_rot,
                        cam_proj=cam_proj,
                        pred_threshold=self.config.lap_threshold
                )

                iou_epoch['surf'] += surface_align.mean().item()
                iou_epoch['occ_iou'] += occ_iou.mean().item()
                iou_epoch['lap'] += lap.mean().item()
                iou_epoch['edge'] += edge.mean().item()

                iou_epoch['area'] += area_variance.mean().item()
                iou_epoch['delta'] += delta_loss.mean().item()
                iou_epoch['surf_chamfer'] += other_chamfer_distance.mean().item()
                iou_epoch['amips'] += amips_energy.mean().item()

                ######################
                pred_points_occ_prob = self.deftet.paste_occ(
                   pred_occ_prob,  condition.clone())
                # import ipdb
                # ipdb.set_trace()
                gt_occ[gt_occ > 0] = 1.0
                gt_occ[gt_occ <= 0] = 0.0
                for pt1, pt2 in zip(gt_occ, pred_points_occ_prob):
                    for t in self.threshold_list:
                        iou_epoch[t] += float((point_cloud_iou(
                            pt1, pt2, thresh=t).item() / float(gt_occ.shape[0])))
                num_batches += 1

            max_iou = 0
            for t in self.threshold_list:
                out_loss = iou_epoch[t] / float(num_batches)
                self.writer.add_scalar('val_iou_%.1f' %
                                       (t), out_loss, self.global_step)
                print(
                        f'[VAL IoU Total] Epoch {self.cur_epoch:03d}, Batch {i:03d} t: {t:1.1f}, iou: {out_loss:3.3f}')
                max_iou = max(max_iou, out_loss)


            self.writer.add_scalar('val_iou_max', max_iou, self.global_step)
            show_list = ['surf', 'occ_iou', 'lap', 'edge', 'surf_chamfer',
                         'boundary', 'area', 'delta', 'amips', ]
            for show_name in show_list:
                self.writer.add_scalar(
                    'val_' + show_name, iou_epoch[show_name] / float(num_batches), self.global_step)
                print('val_' + show_name, iou_epoch[show_name] / float(num_batches))

            self.val_loss.append(max_iou)

    def save(self, step=None):
        save_best = False
        if len(self.val_loss) >= 1 and self.val_loss[-1] >= self.bestval:
            self.bestval = self.val_loss[-1]
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': self.bestval,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_metrics': ['Chamfer'],
            'val_metrics': ['Chamfer'],
            'global_step': self.global_step,
        }
        # Save the recent model/optimizer states
        prefix=''
        if not step is None:
            prefix = '_'+str(step)
        torch.save(self.model.encoder.state_dict(),
                   self.experiment.file_path('encoder'+prefix+'.pth'))
        torch.save(self.model.decoder_occ.state_dict(),
                   self.experiment.file_path('decoder_occ'+prefix+'.pth'))
        torch.save(self.model.decoder_pos.state_dict(),
                   self.experiment.file_path('decoder_pos'+prefix+'.pth'))
        if self.config.use_lap_layer:
            torch.save(self.model.lap_decoder_pos.state_dict(),
                       self.experiment.file_path('lap_decoder_pos'+prefix+'.pth'))
        torch.save(self.optimizer.state_dict(),
                       self.experiment.file_path('recent_optim'+prefix+'.pth'))

        with open(self.experiment.file_path('recent.log'), 'w') as f:
            f.write(json.dumps(log_table))
        print('====== Saved recent model ======>')
        if save_best:
            torch.save(self.model.encoder.state_dict(),
                       self.experiment.file_path('best_encoder.pth'))
            torch.save(self.model.decoder_occ.state_dict(),
                       self.experiment.file_path('best_decoder_occ.pth'))
            torch.save(self.model.decoder_pos.state_dict(),
                       self.experiment.file_path('best_decoder_pos.pth'))

            if self.config.use_lap_layer:
                torch.save(self.model.lap_decoder_pos.state_dict(),
                           self.experiment.file_path('best_lap_decoder_pos.pth'))
            torch.save(self.optimizer.state_dict(),
                           self.experiment.file_path('best_optim.pth'))

            print('====== Overwrote best model ======>')
            print(str(log_table))
            print('============')


def main(experiment, config, state):
   main_worker( config, experiment)

def main_worker(config, experiment):
    timing = None
    train_for_debug = True####
    dataloader_train = create_dataloader(batch_size=config.batch_size, only_chairs= train_for_debug)
    dataloader_val = create_dataloader(batch_size=config.batch_size, train=False, only_chairs= train_for_debug)
    trainer = Engine(timing=timing,
                     config=config,
                     dataloader_train=dataloader_train,
                     dataloader_val=dataloader_val,
                     experiment=experiment)

    epochs = config.epochs
    if config.timing:
        print('NOTE: Number of epochs has been set to 1 due to --timing')
        epochs = 1

    if config.use_lap_layer:
        step = 1
    else:
        step = 5
    epoch = 0
    # trainer.validate_iou()
    trainer.save(epoch * len(trainer.dataloader_train))
    for epoch in range(epochs):

        trainer.train()
        if epoch % step == 0 and epoch != 0:
            torch.cuda.empty_cache()
            trainer.validate_iou()
            trainer.save(epoch * len(trainer.dataloader_train))

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    experiment = Experiment.new(
        default_folder_path=DEFAULT_FOLDER_PATH,
        short_info=SHORT_INFO,
        info=INFO,
        options=OPTIONS,
    )
    experiment.run(main)