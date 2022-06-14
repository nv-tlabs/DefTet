'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
import torch.nn as nn

import numpy as np

# load tet
from prepare_for_wz import read_tetrahedron, tet_to_face_idx, generate_point_adj_idx
from utils_tetsv import get_face_use_occ, get_face_use_occ_color, tet_adj_share, save_tet_face_color, save_tet_face
from cameraop import perspective

import sys
sys.path.append('..')
from config import rootdir
sys.path.append('%s/utils' % rootdir)
from utils_mesh import savemesh, savemeshfweights, savemeshfweightscolor


class Deftet(nn.Module):
    def __init__(self,
                 basefolder,
                 res,
                 coef,
                 feature_dim=4,
                 feature_raw=True,
                 feature_fixed_dim=0,
                 feature_fixed_init=None,
                 neighbourlayer=3):
        super(Deftet, self).__init__()
        r"""
        
        Args:
            basefolder (str): where you store tet files
            res (int): resolution of tet
            coef (float): the tet will be normalized in [-0.5, 0.5], but we time it with a coef to better cover the object
            feature_dim (int): the feature of each vertex, generally 4 for rgba
        """

        # tet resolution
        self.res = res

        # read tetrahedron
        file_name = '%s/cube_%d_tet.tet' % (basefolder, res)
        points_px3, tet_list_tx4, _ = read_tetrahedron(file_name, res=0.02)

        # coef to enlarge the shape
        self.coef = coef
        self.neilayer = neighbourlayer

        ################################################################
        # preprocess the points
        p = points_px3
        pmax = np.max(p, axis=0, keepdims=True)
        pmin = np.min(p, axis=0, keepdims=True)
        pmiddle = (pmax + pmin) / 2
        p = p - pmiddle

        # rotate p around y axis
        angle = 0.0 / 180.0 * np.pi
        roty = np.array([[np.cos(angle), 0, np.sin(angle)], \
                         [0, 1, 0], \
                         [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
        p = np.matmul(roty, p.transpose()).transpose()

        pmax = np.max(p, axis=0)
        pmin = np.min(p, axis=0)
        print('pmax, {}, \npmin, {} \ncoef, {}'.format(pmax, pmin, coef))
        assert pmax.max() <= 0.5 and pmin.min() >= -0.5

        ################################################################
        self.feature_dim = feature_dim
        points_px3 = p
        pointfeat_pxd = np.random.rand(p.shape[0],
                                       feature_dim).astype(np.float32)

        # we initilize feature to be [0, 1) if they are raw rgbd
        # or [-1, 1] if they are high dimension features
        if not feature_raw:
            pointfeat_pxd = pointfeat_pxd * 2 - 1

        ###############################################################
        # texture volume
        # sometimes, we also have fixed features
        # e.g. pretrained features
        # or texture coordinate
        self.features_fixed = feature_fixed_dim > 0
        self.features_fixed_dim = feature_fixed_dim
        if self.features_fixed:
            if feature_fixed_dim == 3 and feature_fixed_init is None:
                # initilize with texture coordinate
                feature_fixed_init = points_px3 * 2 * 0.95

        self.updatevaribale(points_px3,
                            pointfeat_pxd,
                            pointmov_px3=None,
                            pointfeat_fixed_pxd=feature_fixed_init)
        self.updategeometry(tet_list_tx4)
        return

    ################################################################################
    def updatevaribale(self,
                       points_px3,
                       pointfeat_pxd,
                       pointmov_px3=None,
                       pointfeat_fixed_pxd=None):

        #####################################################
        # feed to pytorch
        tfp_px3 = torch.from_numpy(points_px3)
        self.tfpoint_px3 = tfp_px3
        if pointmov_px3 is None:
            self.tfpointmov_px3 = nn.Parameter(torch.zeros_like(tfp_px3))
        else:
            self.tfpointmov_px3 = nn.Parameter(torch.from_numpy(pointmov_px3))

        tfpointfeat_pxd = torch.from_numpy(pointfeat_pxd)
        self.tfpointfeat_pxd = nn.Parameter(tfpointfeat_pxd)

        # fixed features
        # e.g. coordinates
        if self.features_fixed:
            self.tfpointfeat_fixed_pxd = torch.from_numpy(pointfeat_fixed_pxd)
        else:
            self.tfpointfeat_fixed_pxd = None

        self.n_point = self.tfpoint_px3.shape[0]

        return

    def updategeometry(self, tet_list_tx4):

        n_point = self.n_point

        # convert to face (without duplicate face)
        faces_fx3, face_tet_idx_fx2, _ = tet_to_face_idx(n_point,
                                                         tet_list_tx4,
                                                         with_boundary=True)
        # feed to pytorch
        # no need to train
        self.tff_fx3 = torch.from_numpy(faces_fx3)
        self.tftet_tx4 = torch.from_numpy(tet_list_tx4)
        self.tftet2face_fx2 = torch.from_numpy(face_tet_idx_fx2)

        # used in saving
        self.tet_adj, self.tet_neighbour_idx = tet_adj_share(
            tet_list_tx4, n_point)

        ############################################################
        # point neightbour
        point_adj_idx_pxm, point_adj_weights_px1 = generate_point_adj_idx(
            n_point, tet_list_tx4)

        self.tfpoint_adj_idx_pxm = torch.from_numpy(point_adj_idx_pxm).cuda() + 1
        self.tfpoint_adj_weights_px1 = torch.from_numpy(point_adj_weights_px1) + 1e-10

        return

    ################################################
    def sethw(self, height, width, multiplier):

        xidx = torch.arange(start=0,
                            end=width,
                            step=1,
                            dtype=torch.float32,
                            requires_grad=False)
        xidx = (xidx + 0.5) / width * 2.0 - 1.0

        yidx = torch.arange(start=0,
                            end=height,
                            step=1,
                            dtype=torch.float32,
                            requires_grad=False)
        yidx = (yidx + 0.5) / height * 2.0 - 1.0
        yidx = -yidx

        ymap_hxw, xmap_hxw = torch.meshgrid(yidx, xidx)
        xy_hxwx2 = torch.stack([xmap_hxw, ymap_hxw], dim=2)
        xy_px2 = xy_hxwx2.view(-1, 2)

        self.height = height
        self.width = width
        self.multiplier = multiplier
        self.xy_px2 = xy_px2
        return

    def todev(self, dev):
        self.tfpoint_px3 = self.tfpoint_px3.to(dev)
        self.tfpoint_adj_weights_px1 = self.tfpoint_adj_weights_px1.to(dev)
        self.xy_px2 = self.xy_px2.to(dev)
        if self.features_fixed:
            self.tfpointfeat_fixed_pxd = self.tfpointfeat_fixed_pxd.to(dev)
        self.device = dev
        return

    def get_hw(self):
        return self.height, self.width

    def get_point(self, with_coef=False):
        if with_coef:
            return self.coef * (self.tfpoint_px3 + self.tfpointmov_px3)
        else:

            return self.tfpoint_px3 + self.tfpointmov_px3

    def get_mov(self):
        return self.tfpointmov_px3

    def get_feat(self):
        tffeat = self.tfpointfeat_pxd
        if self.features_fixed:
            tffeat = torch.cat([tffeat, self.tfpointfeat_fixed_pxd], dim=1)
        return tffeat

    def get_featlap(self, pointfeat_px3):

        offset_px3 = pointfeat_px3
        #print("offset_px3.device: ", offset_px3.device)
        #offset_1px3 = torch.cat([torch.zeros_like(offset_px3[:1]), offset_px3],
        #                        dim=0)
        offset_1px3 = torch.nn.functional.pad(offset_px3, (0, 0, 1, 0))

        point_adj_idx = self.tfpoint_adj_idx_pxm
        #print("point_adj_idx.device: ", point_adj_idx.device)
        offset_pmx3 = offset_1px3[point_adj_idx.view(-1), :]


        pnum, mnum = self.tfpoint_adj_idx_pxm.shape
        offset_pxmx3 = offset_pmx3.view(pnum, mnum, -1)

        point_adj_weights_px1 = self.tfpoint_adj_weights_px1
        #print("point_adj_weights_px1.device: ", point_adj_weights_px1.device)
        offset_nei_px3 = offset_pxmx3.sum(1) / point_adj_weights_px1

        return torch.nn.functional.mse_loss(offset_nei_px3, offset_px3, reduction='none')

    def get_volume_variance(self,
                            base_area_mask=None,
                            area_normalize=(20, 20),
                            pow=2,
                            center_occ=None):
        def cross_dot_torch(a, b):
            normal = torch.zeros_like(a)
            normal[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
            normal[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
            normal[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            return normal

        def det_m(m_bx3x3):
            a = m_bx3x3[:, 0, :]
            b = m_bx3x3[:, 1, :]
            c = m_bx3x3[:, 2, :]
            det = torch.sum(a * cross_dot_torch(b, c), dim=-1)
            return det

        # prepare
        v_pos_bxnx3 = self.get_point().unsqueeze(0)
        tet_bxfx4 = self.tftet_tx4.unsqueeze(0).to(v_pos_bxnx3.device)

        # https://en.wikipedia.org/wiki/Tetrahedron
        # if pow == 4:
        #     scale = 20
        # else:
        #     scale = 20
        scale = 2  # make it from [-0.5, 0.5] to [-1, 1]

        tet_bxfx4x3 = torch.gather(
            input=v_pos_bxnx3.unsqueeze(2).expand(-1, -1, tet_bxfx4.shape[-1],
                                                  -1),
            index=tet_bxfx4.unsqueeze(-1).expand(-1, -1, -1,
                                                 v_pos_bxnx3.shape[-1]),
            dim=1)

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
        V = -det_m(m) / 6.0
        V = V.reshape(a.shape[0], a.shape[1])
        '''
        mean_v = torch.ones(tet_bxfx4x3.shape[0], 1,
                            dtype=tet_bxfx4x3.dtype, device=tet_bxfx4x3.device)
        '''

        mean_v = torch.mean(V, dim=-1, keepdim=True)
        # n_tet = tet_bxfx4x3.shape[1]
        # mean_v = mean_v * 1.0 * scale * scale * scale / n_tet
        '''
        if pow == 1:
            var_v = torch.sum(torch.abs(V - mean_v), dim=-1)
        else:
            var_v = torch.sum((V - mean_v)**pow, dim=-1)
        return var_v
        '''
        return (V - mean_v).squeeze(0)

    ######################################################################
    def pointweights2tetweights(self, point_weights_px1, tet_list_tx4):
        point_weights_p = point_weights_px1[:, 0]
        tet_weights_tx4 = point_weights_p[tet_list_tx4]
        return tet_weights_tx4

    def tetweights2tetneighbourweights(self, tet_weights_tx4, neilevel=1):

        tet_weights_txk = tet_weights_tx4

        while neilevel > 0:
            neilevel = neilevel - 1
            tnum, knum = tet_weights_txk.shape
            tet_weights_1txk = np.zeros((tnum + 1, knum), dtype=np.float32)
            tet_weights_1txk[1:] = tet_weights_txk

            tet_weights_t4xk = tet_weights_1txk[
                self.tet_neighbour_idx.reshape(-1, ) + 1]
            tet_weights_tx4k = tet_weights_t4xk.reshape(tnum, -1)
            tet_weights_txk = tet_weights_tx4k

        return tet_weights_txk

    def deletetet(self, thres, processfunc):

        tfweights_px1, tfcolors_px3 = processfunc(
            self.get_point(with_coef=True), self.get_feat())
        point_weights_px1 = tfweights_px1.detach().cpu().numpy()

        tet_list_tx4 = self.tftet_tx4.detach().cpu().numpy()
        tet_weights_tx4 = self.pointweights2tetweights(point_weights_px1,
                                                       tet_list_tx4)
        tet_weights_tx64 = self.tetweights2tetneighbourweights(
            tet_weights_tx4, neilevel=self.neilayer)

        from prepare_for_wz import delete_tet
        tet_list_new_Kx4 = delete_tet(tet_list_tx4, tet_weights_tx64, thres)
        if len(tet_list_new_Kx4) == 0:
            tet_list_new_Kx4 = tet_list_tx4

        self.updategeometry(tet_list_new_Kx4)
        return

    ##########################################################
    def tensor2ndarray(self, processfunc=None):
        points_px3 = self.tfpoint_px3.detach().cpu().numpy()
        pointmov_px3 = self.tfpointmov_px3.detach().cpu().numpy()
        pfeat_pxk = self.get_feat().detach().cpu().numpy()
        tet_list_tx4 = self.tftet_tx4.detach().cpu().numpy()

        if processfunc is None:
            point_weights_px1 = None
        else:
            tfweights_px1, tfcolors_px3 = processfunc(
                self.get_point(with_coef=True), self.get_feat())
            point_weights_px1 = tfweights_px1.detach().cpu().numpy()

        return points_px3, pointmov_px3, pfeat_pxk, tet_list_tx4, point_weights_px1

    def subdivision(self, loadpth=None, thres=None, processfunc=None):

        # we will start from fixed saving pth
        # it should be more even
        if loadpth is not None:
            filename = '%s/deftet.pth' % (loadpth, )
            self.load_state_dict(torch.load(filename))

        from prepare_for_wz import generate_subdivision
        points_px3, pointmov_px3, pfeat_pxk, tet_list_tx4, point_weights_px1 = self.tensor2ndarray(
            processfunc)

        if thres is None:
            tet_subdiv = None
        else:
            assert point_weights_px1 is not None
            tet_weights_tx4 = self.pointweights2tetweights(
                point_weights_px1, tet_list_tx4)
            # if all the weights are > big, no need to do subdiv?
            tet_weights_t = np.min(tet_weights_tx4, axis=1)
            tet_subdiv = tet_weights_t < thres

        tet_points_new_Px3, tet_feat_new_Pxk, tet_list_new_Tx4 = \
        generate_subdivision(tet_list_tx4, points_px3, np.concatenate([pfeat_pxk, pointmov_px3], axis=1), tet_subdiv)

        pmov_px3 = tet_feat_new_Pxk[:, -3:]
        pfeat_pxk = tet_feat_new_Pxk[:, :-3]
        if self.features_fixed:
            pfeat_fixed_pxk = pfeat_pxk[:, -self.features_fixed_dim:]
            pfeat_pxk = pfeat_pxk[:, :-self.features_fixed_dim]
        else:
            pfeat_fixed_pxk = None

        self.updatevaribale(tet_points_new_Px3, pfeat_pxk, pmov_px3,
                            pfeat_fixed_pxk)
        self.updategeometry(tet_list_new_Tx4)

    #################################################
    def forward(self,
                impixsample_hxw,
                camrot_bx3x3,
                camtrans_bx3,
                camproj_3x1,
                renderfunc,
                viewpoint=False,
                depth=False,
                istraining=False):
        r"""
        Args:
            cameras_bx3x3 (torch.float32), camera_origin, camera_lookat, camera_up

        Returns:
            new_node_feat: (batch_size, num_nodes, output_dim)
        """

        bs = camrot_bx3x3.shape[0]

        ##########################################################
        tfpoint_px3 = self.get_point(True)
        tfp_1xpx3 = tfpoint_px3.unsqueeze(0)
        tfp_bxpx3 = tfp_1xpx3.repeat(bs, 1, 1)

        tfpointfeat_pxd = self.get_feat()
        tfpfeat_1xpxd = tfpointfeat_pxd.unsqueeze(0)
        tfpfeat_bxpxd = tfpfeat_1xpxd.repeat(bs, 1, 1)

        # camera
        tfpointworld_bxpx3 = tfp_bxpx3
        tfcameras = [camrot_bx3x3, camtrans_bx3, camproj_3x1]
        vertices_camera_bxpx3, vertices_image_bxpx2 = perspective(
            tfpointworld_bxpx3, tfcameras)

        # warp viewdir and depth
        # camer dir?
        if viewpoint:
            tfcam_world_bx3x1 = tfcameras[1]
            viewdir_bxpx3 = tfcam_world_bx3x1.view(-1, 1,
                                                   3) - tfpointworld_bxpx3
            viewlen_bxpx1 = torch.sqrt((viewdir_bxpx3**2).sum(dim=2,
                                                              keepdim=True))
            viewdir_bxpx3 = viewdir_bxpx3 / (viewlen_bxpx1 + 1e-10)
            tfpfeat_bxpxd = torch.cat([viewdir_bxpx3, tfpfeat_bxpxd], dim=2)

        if depth:
            depth_camera_bxpx1 = vertices_camera_bxpx3[:, :, 2:3]
            tfpfeat_bxpxd = torch.cat([depth_camera_bxpx1, tfpfeat_bxpxd],
                                      dim=2)

        #########################################################
        # multiplier
        xy_px2 = self.xy_px2[impixsample_hxw.view(-1, )]
        xy_bxpx2 = xy_px2.unsqueeze(0).repeat(bs, 1, 1)
        xydep_bxpx2 = torch.zeros_like(xy_bxpx2)
        xydep_bxpx2[:, :, 0] = -1000

        imcolor_bxpx3, immask_bxpx1, imdep_bxpx1 = renderfunc(
            xy_bxpx2 * self.multiplier,
            xydep_bxpx2,
            vertices_camera_bxpx3,
            vertices_image_bxpx2 * self.multiplier,
            tfpfeat_bxpxd,
            self.tff_fx3,
            viewdir=viewpoint,
            depth=depth,
            istraining=istraining)

        if depth:
            return imcolor_bxpx3, immask_bxpx1, imdep_bxpx1
        else:
            return imcolor_bxpx3, immask_bxpx1

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        models = nn.Module.state_dict(self,
                                      destination=destination,
                                      prefix=prefix,
                                      keep_vars=keep_vars)
        models['points'] = self.tfpoint_px3
        models['tets'] = self.tftet_tx4
        models['feat_fixed'] = self.tfpointfeat_fixed_pxd
        return models

    def load_state_dict(self, state_dict, strict=True):
        models = state_dict
        points_px3 = models['points'].detach().cpu().numpy()
        tet_list_tx4 = models['tets'].detach().cpu().numpy()
        pointfeat_pxd = models['tfpointfeat_pxd'].detach().cpu().numpy()
        pointfeat_fixed_pxd = models['feat_fixed'].detach().cpu().numpy()
        pointmov_px3 = models['tfpointmov_px3'].detach().cpu().numpy()

        self.updatevaribale(points_px3, pointfeat_pxd, pointmov_px3,
                            pointfeat_fixed_pxd)
        self.updategeometry(tet_list_tx4)
        return

    def saveobj(self, savedir, prefix, processfunc):

        tfpoint_px3 = self.get_point(True)
        pointsnp_px3 = tfpoint_px3.detach().cpu().numpy()
        facenp_fx3 = self.tff_fx3.detach().cpu().numpy()

        tfweights_px1, tfcolors_px3 = processfunc(self.get_point(True),
                                                  self.get_feat())
        weightsnp_px1 = tfweights_px1.detach().cpu().numpy()
        colorsnp_px3 = tfcolors_px3.detach().cpu().numpy()
        colorsnp_px3 = colorsnp_px3[:, ::-1]

        tet_list_tx4 = self.tftet_tx4.detach().cpu().numpy()

        tet_p = pointsnp_px3[tet_list_tx4.reshape(-1, ), :]
        tet_p_1xtx4x3 = tet_p.reshape(1, -1, 4, 3)
        tet_pcolor = colorsnp_px3[tet_list_tx4.reshape(-1, ), :]
        tet_pcolor_px1xtx4x3 = tet_pcolor.reshape(1, -1, 4, 3)

        tet_occ = weightsnp_px1[tet_list_tx4.reshape(-1, ), :]
        tet_occ_tx1 = tet_occ.reshape(-1, 4, 1).max(1)

        # 3 colors
        tet2face_idx_fx2 = self.tftet2face_fx2.detach().cpu().numpy()
        tetocc_1tx1 = np.concatenate(
            [np.zeros_like(tet_occ_tx1[:1, :]), tet_occ_tx1], axis=0)
        focc_1 = tetocc_1tx1[tet2face_idx_fx2[:, 0] + 1, :]
        focc_2 = tetocc_1tx1[tet2face_idx_fx2[:, 1] + 1, :]
        focc_fx1 = focc_1 * (1 - focc_2) + focc_2 * (1 - focc_1)

        for thres in [0.005, 0.05, 0.15, 0.25]:
            '''
            filename = '%s/face-geo-%s-thres-%.3f.obj' % (savedir, prefix,
                                                          thres)
            savemeshfweights(pointsnp_px3, focc_fx1, facenp_fx3, filename,
                             thres)
            filename = '%s/face-color-%s-thres-%.3f.obj' % (savedir, prefix,
                                                            thres)
            savemeshfweightscolor(pointsnp_px3, focc_fx1, facenp_fx3, filename,
                                  thres, colorsnp_px3)
            
            '''
            filename = '%s/tet-geo-%s-thres-%.3f.obj' % (savedir, prefix,
                                                         thres)
            tmpp = get_face_use_occ(tet_p_1xtx4x3, tet_occ_tx1, self.tet_adj,
                                    thres)
            save_tet_face(tmpp[0], f_name=filename)

            filename = '%s/tet-color-%s-thres-%.3f.obj' % (savedir, prefix,
                                                           thres)
            tmpp, tmpc = get_face_use_occ_color(tet_p_1xtx4x3,
                                                tet_pcolor_px1xtx4x3,
                                                tet_occ_tx1, self.tet_adj,
                                                thres)
            save_tet_face_color(tmpp[0], tmpc[0], f_name=filename)
