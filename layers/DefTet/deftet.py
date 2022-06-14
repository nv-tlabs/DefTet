'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import numpy as np
import torch.nn as nn
import torch
from utils import tet_utils
from utils import mesh_utils
from utils import matrix_utils
import kaolin as kal
from layers.DefTet.check_condition_tetrahedron_base.utils import check_condition_f_base

EPS = 1e-10


class DefTet(nn.Module):
    def __init__(self,
                 device=None,

                 ):
        super(DefTet, self).__init__()
        self.pow = 4
        self.device = device
        self.features_fixed = False
        self.z_window_radius = 0.025
        self.inverse_v = None

    def check_tet_inside_sdfs(self,
                              tet_bxfx4x3,
                              mesh_list):
        batch_size = tet_bxfx4x3.shape[0]
        scale = 1
        verts = mesh_list[0]
        faces = mesh_list[1]
        with torch.no_grad():
            occupancy = []
            # import ipdb
            # ipdb.set_trace()
            for v, f, tet_fx4x3 in zip(verts, faces, tet_bxfx4x3):
                center = torch.mean(tet_fx4x3, dim=1) * scale
                result = kal.ops.mesh.check_sign(v, f[0], center.unsqueeze(dim=0), hash_resolution=512)
                occupancy.append(result.unsqueeze(-1))
            occupancy = torch.cat(occupancy, dim=0).float()
        return occupancy

    def forward_surface_align(self,
                              vertice_pos,
                              point_pos_bxpx3,
                              tetrahedron_bxfx4=None,
                              mesh_list=None,
                              gt_surface_points=None,
                              tet_face_bxfx3=None,
                              inference=False,
                              pred_occ=None,
                              tet_face_tet_bx4fx2=None,
                              save=False,
                              save_name=None,
                                inference_threshold=0.4
                              ):
        tetrahedron_bxfx4 = tetrahedron_bxfx4.long()
        tet_bxfx4x3 = torch.gather(input=vertice_pos.unsqueeze(2).expand(-1, -1, tetrahedron_bxfx4.shape[-1], -1),
                                       index=tetrahedron_bxfx4.unsqueeze(-1).expand(-1, -1, -1, vertice_pos.shape[-1]),
                                       dim=1)
        center_occ = self.check_tet_inside_sdfs(tet_bxfx4x3, mesh_list)

        boundary = self.get_boundary_index(tet_face_bxfx3[0], tet_face_tet_bx4fx2[0], center_occ.squeeze(dim=-1)) ######
        if save:
            for idx, f in enumerate(boundary):

                if idx > 4: break
                f = torch.gather(input=vertice_pos[idx].unsqueeze(dim=-2).expand(-1, 3, -1),
                                 index=boundary[idx].unsqueeze(dim=-1).expand(-1, -1, 3),
                                 dim=0)
                mesh_utils.save_tet_face(
                    f.data.cpu().numpy(), save_name + '_device_%d_%d.obj' % (torch.cuda.current_device(), idx))

        volume_variance = self.volume_variance(tet_bxfx4x3, pow=self.pow)
        amips_energy = self.amips_energy(tet_bxfx4x3, self.inverse_v.clone().to(tet_bxfx4x3.device), center_occ=None)

        sum_chamfer_distance = 0.0
        sum_analytic_distance = 0.0
        sum_normal_loss = 0.0

        for i in range(vertice_pos.shape[0]):
            ## This one can not be batched as we have different surface per shape
            chamfer_distance, analytic_distance, normal_loss = \
                self.forward(
                    v_pos_bxnx3=vertice_pos[i:i+1],
                    tet_bxfx4=tetrahedron_bxfx4[i:i+1],
                    boundary_bxfx3=boundary[i].unsqueeze(dim=0),
                    gt_surface_point=gt_surface_points[i:i+1],
                    inverse_offset=self.inverse_v,
                    tet_bxfx4x3=tet_bxfx4x3[i:i+1],
                    calculate_amips_volume=False
                )
            sum_chamfer_distance += (chamfer_distance / vertice_pos.shape[0])
            sum_analytic_distance += (analytic_distance / vertice_pos.shape[0])
            sum_normal_loss += (normal_loss / vertice_pos.shape[0])

        edge = self.edge_length(tet_bxfx4x3, pow=self.pow)

        lap_v_loss = torch.zeros_like(sum_normal_loss)

        center_occ = center_occ.squeeze(-1)
        if inference:
            assert point_pos_bxpx3 is not None, 'point_pos_bxpx3 not given'
            condition = check_condition_f_base(tet_bxfx4x3, point_pos_bxpx3)
            pred_occ = (pred_occ > inference_threshold).float()
            pred_surface_face = self.get_boundary_index(tet_face_bxfx3[0], tet_face_tet_bx4fx2[0], pred_occ)###
            return (amips_energy,
                    edge,
                    volume_variance,
                    sum_analytic_distance,
                    sum_normal_loss,
                    center_occ,
                    condition,
                    boundary, pred_surface_face, sum_chamfer_distance)
        else:
            return (amips_energy,
                    edge,
                    volume_variance,
                    sum_analytic_distance,
                    sum_normal_loss,
                    center_occ,
                    boundary, sum_chamfer_distance, lap_v_loss)

    def paste_occ(self, pred_tet_occ, condition):
        condition[condition < 0] = 0
        pred_occ = torch.gather(
            input=pred_tet_occ, index=condition.long().squeeze(-1), dim=1)
        return pred_occ

    def forward(self,
                v_pos_bxnx3=None,
                tet_bxfx4=None,
                boundary_bxfx3=None,
                gt_surface_point=None,
                inverse_offset=None,
                tet_bxfx4x3=None,
                calculate_amips_volume=True):
        # import ipdb
        # ipdb.set_trace()
        if tet_bxfx4x3 is None:
            tet_pos_bxfx4x3 = torch.gather(input=v_pos_bxnx3.unsqueeze(2).expand(-1, -1, tet_bxfx4.shape[-1], -1),
                                           index=tet_bxfx4.unsqueeze(-1).expand(-1, -1, -1, v_pos_bxnx3.shape[-1]),
                                           dim=1)
        else:
            tet_pos_bxfx4x3 = tet_bxfx4x3
        if calculate_amips_volume:
            area_variance = self.volume_variance(tet_pos_bxfx4x3, pow=self.pow)
            if not inverse_offset is None:

                amips_energy = self.amips_energy(tet_pos_bxfx4x3, inverse_offset, center_occ=None)
            else:
                amips_energy = torch.zeros_like(area_variance)

        if boundary_bxfx3.shape[1] == 0:
            one_loss = torch.ones(1, device=boundary_bxfx3.device)
            if calculate_amips_volume:
                return one_loss, one_loss, one_loss, area_variance, amips_energy, tet_pos_bxfx4x3
            return one_loss, one_loss, one_loss

        surface_pos = torch.gather(input=v_pos_bxnx3.unsqueeze(2).expand(-1, -1, boundary_bxfx3.shape[-1], -1),
                                     index=boundary_bxfx3.unsqueeze(-1).expand(-1, -1, -1, v_pos_bxnx3.shape[-1]),
                                     dim=1)

        normal_loss = mesh_utils.get_surface_normal_loss(v_pos_bxnx3, boundary_bxfx3)

        pred_surface_point = mesh_utils.sample_surf_point_batch(surface_pos, 20)
        pred_surface_point = pred_surface_point.reshape(pred_surface_point.shape[0], -1, 3)
        gt_surface_point = gt_surface_point.reshape(gt_surface_point.shape[0], -1, 3)
        chamfer_distance = mesh_utils.point_point_distance(pred_surface_point, gt_surface_point)

        analytic_distance = mesh_utils.point_mesh_distance(gt_surface_point, surface_pos)
        chamfer_distance = chamfer_distance.mean(dim=-1)
        analytic_distance = analytic_distance.mean(dim=-1).mean(dim=-1)
        if calculate_amips_volume:
            return chamfer_distance, analytic_distance, normal_loss, area_variance, amips_energy, tet_pos_bxfx4x3
        return chamfer_distance, analytic_distance, normal_loss

    def get_boundary_index(self, tet_face_fx3, tet_idx_fx2, occ_bxn):
        tet_face_occ_bxfx2 = torch.gather(input=occ_bxn, index=tet_idx_fx2.reshape(-1).unsqueeze(0).expand(occ_bxn.shape[0], -1),
                                          dim=1).reshape(occ_bxn.shape[0], -1, 2)
        tet_face_occ_bxf = tet_face_occ_bxfx2.sum(dim=-1)
        tet_occ = [b[t == 1] for b,t in zip(tet_face_occ_bxfx2, tet_face_occ_bxf)]
        change_idx = [(t[:,0] == 1).unsqueeze(-1) for t in tet_occ]
        boundary_index = [tet_face_fx3[t==1] for t in tet_face_occ_bxf]
        change_boundary = [b.flip(dims=[1]) for b in boundary_index]
        boundary_index = [b * (~c) + cb * c for b,c,cb in zip(boundary_index, change_idx, change_boundary)]
        return boundary_index

    def get_internal_index(self, tet_face_fx3, tet_idx_fx2, occ_bxn):
        tet_face_occ_bxfx2 = torch.gather(input=occ_bxn, index=tet_idx_fx2.reshape(-1).unsqueeze(0).expand(occ_bxn.shape[0], -1),
                                          dim=1).reshape(occ_bxn.shape[0], -1, 2)

        tet_face_occ_bxf = tet_face_occ_bxfx2.sum(dim=-1)
        boundary_index = [tet_face_fx3[t == 2] for t in tet_face_occ_bxf]
        return boundary_index

    def my_inverse(self, T):
        max_batch = 65535
        n_batch = T.shape[0]
        n_split = int(n_batch / max_batch) + 1
        result_list = []
        mask_list = []

        for i in range(n_split - 1):
            # remove non-singular matrix
            tmp_m = T[i * max_batch: (i+1) * max_batch]
            det_m = torch.abs(torch.det(tmp_m)) < 1e-10
            det_m = det_m.float()
            iden_m = torch.eye(tmp_m.shape[-1], dtype=torch.float,
                               device=tmp_m.device).unsqueeze(0).expand(tmp_m.shape[0], -1, -1)
            tmp_m = tmp_m * (1 - det_m.unsqueeze(-1).unsqueeze(-1)) + \
                iden_m * det_m.unsqueeze(-1).unsqueeze(-1)
            result_list.append(torch.inverse(tmp_m))
            mask_list.append(det_m)

        tmp_m = T[(n_split-1)*max_batch:]
        det_m = torch.abs(torch.det(tmp_m)) < 1e-10
        det_m = det_m.float()
        iden_m = torch.eye(tmp_m.shape[-1], dtype=torch.float, device=tmp_m.device).unsqueeze(0).expand(tmp_m.shape[0],
                                                                                                        -1, -1)
        tmp_m = tmp_m * (1 - det_m.unsqueeze(-1).unsqueeze(-1)) + \
            iden_m * det_m.unsqueeze(-1).unsqueeze(-1)
        result_list.append(torch.inverse(tmp_m))
        mask_list.append(det_m)
        return torch.cat(result_list, dim=0), 1 - torch.cat(mask_list, dim=0)

    def save_mesh(self, tet, tet_fea, save_name='debug/tmp.obj'):
        new_tet = tet[tet_fea.squeeze() > 0.5]
        mesh_utils.save_tet_simple(new_tet.data.cpu().numpy(), save_name)

    def volume_variance(self, tet_bxfx4x3, base_area_mask=None, area_normalize=(20, 20), pow=2, center_occ=None):
        # https://en.wikipedia.org/wiki/Tetrahedron
        scale = 1
        A = tet_bxfx4x3[:, :, 0, :] * scale
        B = tet_bxfx4x3[:, :, 1, :] * scale
        C = tet_bxfx4x3[:, :, 2, :] * scale
        D = tet_bxfx4x3[:, :, 3, :] * scale

        # V = |(a - d) * ((b - d) x (c - d))| / 6
        a = A - D
        b = B - D
        c = C - D
        m = torch.cat([a.unsqueeze(2), b.unsqueeze(2), c.unsqueeze(2)], dim=2)
        m = m.reshape(-1, 3, 3)
        V = - matrix_utils.det_m(m) / 6.0
        V = V.reshape(a.shape[0], a.shape[1])
        # if (V < 0).float().sum() > 0:
        #     print('Inverted Tet Detected: %.1f'%(V < 0).float().sum().item())

        mean_v = torch.mean(V, dim=-1, keepdim=True)
        if pow == 1:
            var_v = torch.sum(torch.abs(V - mean_v), dim=-1)
        else:
            var_v = torch.sum((V - mean_v) ** pow, dim=-1)
        return var_v


    def amips_energy(self, tet_bxfx4x3, inverse_v, scale=20,center_occ=None, square=False):
        # scale for numerical stability
        n_batch = tet_bxfx4x3.shape[0]
        A = tet_bxfx4x3[:, :, 0, :].unsqueeze(2) * scale
        B = tet_bxfx4x3[:, :, 1, :].unsqueeze(2) * scale
        C = tet_bxfx4x3[:, :, 2, :].unsqueeze(2) * scale
        D = tet_bxfx4x3[:, :, 3, :].unsqueeze(2) * scale

        offset_vec = torch.cat([B - A, C - A, D - A], dim=2)
        inverse_v = inverse_v.unsqueeze(0).expand_as(offset_vec)
        offset_vec = offset_vec.reshape(-1, 3, 3)
        inverse_v = inverse_v.reshape(-1, 3, 3)
        Jacobian = torch.bmm(offset_vec, inverse_v)
        trace = torch.sum(torch.sum(Jacobian ** 2, dim=-1), dim=-1)

        det = matrix_utils.det_m(Jacobian)
        pos_det = (det >= 0.0).float()
        bottom_det = torch.pow(torch.pow(det, 2) + EPS, -1.0 / 3.0)
        energy = trace * bottom_det * pos_det
        energy = energy.reshape(n_batch, -1)
        if square:
            energy = energy ** 2
        # neg_tet = (det < 0.0).float().reshape(n_batch, -1)
        # if neg_tet.sum(dim=-1)[0] > 0:
        #     print('Inverted Tet Detected: %.1f'%neg_tet.sum().item())
        # if not center_occ is None:
        #     center_occ = center_occ.reshape(n_batch, -1)
        #     energy = energy * center_occ.detach()
            # neg_tet = (det < 0.0).float().reshape(n_batch, -1)
            # if torch.sum(neg_tet) > 0:
            #     print('Inverted Tet: %.5f, %.10f'%(neg_tet.sum().item() / n_batch, (neg_tet.sum().item() / center_occ.sum().item()) / n_batch ))
            # return torch.sum(energy, dim=-1) / center_occ.sum(dim=-1)
        return torch.mean(energy, dim=-1)

    def tet_inverse_v(self, init_tet_pos, init_tet_fx4, scale=20):
        vertice_pos = init_tet_pos.unsqueeze(0).float()
        tetrahedron_bxfx4 = init_tet_fx4.unsqueeze(0)
        n_batch = vertice_pos.shape[0]
        gather_input = vertice_pos.unsqueeze(2).expand(
            n_batch, vertice_pos.shape[1], 4, 3)
        gather_index = tetrahedron_bxfx4.unsqueeze(-1).expand(
            n_batch, tetrahedron_bxfx4.shape[1], 4, 3).long()
        tet_bxfx4x3 = torch.gather(
            input=gather_input, dim=1, index=gather_index)

        A = tet_bxfx4x3[:, :, 0, :].unsqueeze(2) * scale
        B = tet_bxfx4x3[:, :, 1, :].unsqueeze(2) * scale
        C = tet_bxfx4x3[:, :, 2, :].unsqueeze(2) * scale
        D = tet_bxfx4x3[:, :, 3, :].unsqueeze(2) * scale

        offset_vec = torch.cat([B - A, C - A, D - A], dim=2)
        inverse_v = self.my_inverse(offset_vec.squeeze())[0]
        return inverse_v

    def edge_length(self, tet_bxfx4x3, pow=2):
        if pow == 4:
            scale = 20
        else:
            scale = 20

        A = tet_bxfx4x3[:, :, 0, :] * scale
        B = tet_bxfx4x3[:, :, 1, :] * scale
        C = tet_bxfx4x3[:, :, 2, :] * scale
        D = tet_bxfx4x3[:, :, 3, :] * scale
        a = torch.sum((A - D) ** pow, dim=-1).sum(dim=-1)
        b = torch.sum((B - D) ** pow, dim=-1).sum(dim=-1)
        c = torch.sum((C - D) ** pow, dim=-1).sum(dim=-1)
        d = torch.sum((A - B) ** pow, dim=-1).sum(dim=-1)
        e = torch.sum((A - C) ** pow, dim=-1).sum(dim=-1)
        f = torch.sum((B - C) ** pow, dim=-1).sum(dim=-1)

        sum_edge = a + b + c + d + e + f
        return sum_edge / (6 * tet_bxfx4x3.shape[1])

    def laplacian_sparse(self, offset, adj):
        nei_move = matrix_utils.sparse_batch_matmul(adj, offset)
        laploss = torch.sum((nei_move - offset) ** 2, dim=-1).sum(dim=-1)
        return laploss