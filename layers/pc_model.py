'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from torch import nn
import torch.nn.functional as F
from layers.pv_module import functional as pv_F
import torch.distributions as dist
import torch
from layers.pv_utils import create_mlp_components, create_pointnet_components
from layers.gcn_decoder import GCNMLPDecoder
from layers.disn import DISNDecoder, DISNEncoder

# Make sure they have the same interface
class DeformableTetNetwork(nn.Module):
    def __init__(self,
                 device,
                 scale_pos=False,
                 train_def=True,
                 point_cloud=False,
                 point_adj_sparse=None,
                 use_graph_attention=False,
                 upscale=False,
                 use_two_encoder=False,
                 timing=False,
                 use_lap_layer=False,
                 use_disn=False,
                 scale_pvcnn=False,
                 resize_input_shape=True,
                 resize_local_feature_shape=True,
                 predict_color=False,
                 img_in_channels=3,
                 ):

        super(DeformableTetNetwork, self).__init__()
        self.point_cloud = point_cloud
        self.device = device
        self.scale_pos = scale_pos
        self.train_def = train_def
        self.upscale = upscale
        self.use_two_encoder = use_two_encoder
        self.timing = timing
        self.use_lap_layer = use_lap_layer
        self.use_disn = use_disn
        self.scale_pvcnn = scale_pvcnn
        self.blocks = ((64, 1, 32), (128, 2, 16), (512, 1, 8))
        self.predict_color = predict_color

        if self.point_cloud:
            if self.use_two_encoder:
                layers, channels_point, concat_channels_point = create_pointnet_components(
                    blocks=self.blocks, in_channels=3, with_se=False, normalize=False,
                    width_multiplier=1, voxel_resolution_multiplier=1, scale_pvcnn=scale_pvcnn
                )
                encoder_1 = nn.ModuleList(layers)
                layers, channels_point, concat_channels_point = create_pointnet_components(
                    blocks=self.blocks, in_channels=3, with_se=False, normalize=False,
                    width_multiplier=1, voxel_resolution_multiplier=1, scale_pvcnn=scale_pvcnn
                )
                encoder_2 = nn.ModuleList(layers)
                self.encoder = nn.ModuleList([encoder_1, encoder_2]).to(device)
            else:
                layers, channels_point, concat_channels_point = create_pointnet_components(
                    blocks=self.blocks, in_channels=3, with_se=False, normalize=False,
                    width_multiplier=1, voxel_resolution_multiplier=1, scale_pvcnn=scale_pvcnn
                )
                self.encoder = nn.ModuleList(layers).to(self.device)  # PV encoder, we only need the voxel feature from it :)
        else:
            encoder_1 = DISNEncoder(image_size=64, local_feature_size=64, resize_input_shape=resize_input_shape,
                                    resize_local_feature=resize_local_feature_shape, in_channels=img_in_channels).to(self.device)
            encoder_2 = DISNEncoder(image_size=64, local_feature_size=64, resize_input_shape=resize_input_shape,
                                    resize_local_feature=resize_local_feature_shape, in_channels=img_in_channels).to(self.device)
            concat_channels_point = 1024
            self.encoder = nn.ModuleList([encoder_1, encoder_2]).to(device)

        self.pos_encoder = None
        pos_input_dim = self.pos_encoder.get_output_dims() if self.pos_encoder else 3

        if not self.point_cloud:
            self.decoder_pos = DISNDecoder(3, batch_norm=True).to(self.device)
        else:
            self.decoder_pos = GCNMLPDecoder(
                    input_dim=pos_input_dim + concat_channels_point,
                    gcn_hidden_dims=[256, 256, 128],
                    mlp_hidden_dims=[128, 0.2, 64, 3],
                    output_dim=3,
                    adj_sparse=point_adj_sparse,
                    use_attention=use_graph_attention,
                    require_latent=False,
                    use_learned_def_mask=False,
                ).to(self.device)

        if not self.point_cloud:
            self.decoder_occ = DISNDecoder(1, batch_norm=True).to(self.device)
        else:
            occ_input_dim = self.pos_encoder.get_output_dims() if self.pos_encoder else 3
            layers, _ = create_mlp_components(in_channels=(occ_input_dim + concat_channels_point),
                                              out_channels=[256, 0.2, 256, 0.2, 128, 0.2, 64, 1],
                                              classifier=True, dim=2, width_multiplier=1)
            self.decoder_occ = nn.Sequential(*layers).to(self.device)

        if self.upscale:
            self.upsample = nn.Upsample(32, mode='trilinear')

        if self.use_lap_layer:
            if not self.point_cloud:
                self.lap_decoder_pos =DISNDecoder(3, batch_norm=True).to(self.device)
            else:
                self.lap_decoder_pos = GCNMLPDecoder(
                        input_dim=pos_input_dim + concat_channels_point,
                        gcn_hidden_dims=[256, 256, 128],
                        mlp_hidden_dims=[128, 0.2, 64, 3],
                        output_dim=3,
                        adj_sparse=point_adj_sparse,
                        use_attention=use_graph_attention,
                        require_latent=False,
                        use_learned_def_mask=False,
                    ).to(self.device)
            for p in self.decoder_pos.parameters():
                p.requires_grad = False
            for p in self.decoder_occ.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.decoder_occ.eval()
            self.encoder.eval()
            self.decoder_pos.eval()

    def train(self, mode=True):
        if self.use_lap_layer:
            self.lap_decoder_pos.train()
        else:
            self.encoder.train()
            self.decoder_pos.train()
            self.decoder_occ.train()

    def eval(self, mode=True, test=False):
        if self.use_lap_layer:
            self.lap_decoder_pos.eval()
        else:
            self.encoder.eval()
            self.decoder_pos.eval()
            self.decoder_occ.eval()

    def base_encode_inputs(self, inputs, encoder):
        # inputs: N_batch x N_point x 3
        features = inputs.permute(0, 2, 1) * 2
        coords = features[:, :3, :]
        out_features_list = []
        voxel_feature_list = []
        for i in range(len(encoder)):
            features, _, voxel_feature = encoder[i]((features, coords))
            out_features_list.append(features)
            voxel_feature_list.append(voxel_feature)
        if self.upscale:
            voxel_feature_list = [self.upsample(v) for v in voxel_feature_list]
            voxel_feature_list = torch.cat(voxel_feature_list, dim=1)
            voxel_feature_list = [voxel_feature_list]
        return voxel_feature_list

    def encode_images(self, inputs):
        img_feat_1 = self.encoder[0](inputs)
        img_feat_2 = self.encoder[1](inputs)
        return [img_feat_1, img_feat_2]

    def encode_inputs(self, inputs):
        if not self.point_cloud:
            return self.encode_images(inputs)

        if not self.use_two_encoder:
            return self.base_encode_inputs(inputs, self.encoder)

        voxel_feature_list_1 = self.base_encode_inputs(inputs, self.encoder[0])
        voxel_feature_list_2 = self.base_encode_inputs(inputs, self.encoder[1])

        return [voxel_feature_list_1, voxel_feature_list_2]

    def sample_f(self, point_pos, c_list, cam_pos=None,
                                          cam_rot=None,
                                          cam_proj=None):
        if self.point_cloud:
            point_pos = point_pos + 0.5
            point_pos = point_pos.permute(0, 2, 1)
            devoxel_features_list = []
            for c in c_list:
                r = c.shape[-1]
                norm_coords = torch.clamp(point_pos * r, 0, r - 1)
                devoxel_features = pv_F.trilinear_devoxelize(c, norm_coords, r, self.training)
                devoxel_features_list.append(devoxel_features)
            return torch.cat(devoxel_features_list, dim=1)

    def decode_pos(self, p, z, c, init_pos_mask=None,
                   cam_pos=None,
                   cam_rot=None,
                   cam_proj=None):
        pos = p
        if not self.train_def:
            pos_delta = torch.zeros_like(p)
            ori_pos_delta = pos_delta
        else:
            decoder = self.decoder_pos
            pos_feature = self.sample_f(p, c, cam_pos=cam_pos,
                                                cam_rot=cam_rot,
                                                cam_proj=cam_proj)


            pos_feature = torch.cat([pos_feature, pos.permute(0, 2, 1)], dim=1)

            if self.use_disn:
                pos_delta = decoder(pos_feature.permute(0, 2, 1), None, c[-1])
            else:
                pos_delta = decoder(pos_feature)
            pos_delta = pos_delta * 0.1
            pos_delta = pos_delta.permute(0, 2, 1)
            ori_pos_delta = pos_delta

        if self.scale_pos:
            scale = 0.2
            pos_delta = torch.sigmoid(pos_delta) * scale - scale / 2

        if init_pos_mask is not None:
            pos_delta = pos_delta * init_pos_mask
        if not self.train_def:
            pos_delta.zero_()
        pos = p + pos_delta

        if self.use_lap_layer:

            pos_feature = self.sample_f(p, c, cam_pos=cam_pos,
                                                cam_rot=cam_rot,
                                                cam_proj=cam_proj)
            pos_feature = torch.cat([pos_feature, pos.permute(0, 2, 1)], dim=1)

            if self.use_disn:
                lap_pos_delta = self.lap_decoder_pos(pos_feature.permute(0, 2, 1), None, c[-1])
            else:
                lap_pos_delta = self.lap_decoder_pos(pos_feature)

            lap_pos_delta = lap_pos_delta * 0.1

            lap_pos_delta = lap_pos_delta.permute(0, 2, 1)

            if self.scale_pos:
                scale = 0.2
                lap_pos_delta = torch.sigmoid(lap_pos_delta) * scale - scale / 2

            if init_pos_mask is not None:
                lap_pos_delta = lap_pos_delta * init_pos_mask
            lap_pos = pos + lap_pos_delta
            return lap_pos_delta, lap_pos, ori_pos_delta, pos

        return pos_delta, pos, ori_pos_delta

    def get_normal(self, A, B, C):
        a = B - A
        b = C - A
        normal = torch.zeros_like(a)
        normal[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
        normal[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
        normal[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        normal = normal / (torch.sqrt(torch.sum(normal ** 2, dim=-1).unsqueeze(-1) + 1e-10))
        return normal


    def decode_occ(self, pos, z, c, tet_bxfx4=None, use_mask=True,
                   cam_pos=None,
                   cam_rot=None,
                   cam_proj=None,
                   ):
        n_batch = pos.shape[0]

        gather_input = pos.unsqueeze(2).expand(n_batch, pos.shape[1], 4, 3)
        gather_index = tet_bxfx4.unsqueeze(-1).expand(
            n_batch, tet_bxfx4.shape[1], 4, 3).long()
        tet_bxfx4x3 = torch.gather(
            input=gather_input, dim=1, index=gather_index)
        center_pos = torch.mean(tet_bxfx4x3, dim=2)

        center_idx = torch.arange(0, center_pos.shape[1], step=1, device=center_pos.device, dtype=torch.long)
        if use_mask:
            center_idx = torch.randperm(center_pos.shape[1], device=center_pos.device)
            n_select = 10000
            if center_pos.shape[1] < n_select:
                n_select = center_pos.shape[1]
            center_idx = center_idx[:n_select]
            gather_index = center_idx.unsqueeze(
                0).unsqueeze(-1).expand(n_batch, n_select, 3)
            gather_input = center_pos
            masked_center_pos = torch.gather(
                input=gather_input, dim=1, index=gather_index)
        else:
            masked_center_pos = center_pos

        if self.pos_encoder:
            masked_center_pos = self.pos_encoder(masked_center_pos)


        occ_feature = self.sample_f(masked_center_pos, c, cam_pos=cam_pos,
                                            cam_rot=cam_rot,
                                            cam_proj=cam_proj)

        occ_feature = torch.cat([occ_feature, masked_center_pos.permute(0, 2, 1)], dim=1)
        if self.use_disn:
            logits = self.decoder_occ(occ_feature.permute(0, 2, 1), None, c[-1])
            logits = logits.squeeze(dim=1)
        else:
            logits = self.decoder_occ(occ_feature)
            logits = logits.squeeze(dim=1)
        p_r = dist.Bernoulli(logits=logits)
        return p_r, center_idx

    def decode_occ_with_idx(self, pos, z, c, tet_bxfx4):
        def decode_idx(idx, i_batch):
            gather_index = idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, 4)
            gather_input = tet_bxfx4[i_batch].unsqueeze(0)
            sample_tet_bxkx4 = torch.gather(
                input=gather_input, dim=1, index=gather_index)
            logits, _ = self.decode_occ(
                    pos[i_batch].unsqueeze(0),
                    z[i_batch].unsqueeze(0),
                    c[i_batch].unsqueeze(0),
                    sample_tet_bxkx4,
                    use_mask=False,
                    )
            return logits
        return decode_idx

    def split_decode_occ(self, pos, z, c, tet_bxfx4, cam_pos=None,
                   cam_rot=None,
                   cam_proj=None):
        n_batch = pos.shape[0]
        max_split = int(10000 * 10 / n_batch)
        n_f = tet_bxfx4.shape[1]
        n_split = int(n_f / max_split)
        pred_prob_list = []

        for i in range(n_split):
            pred, _ = self.decode_occ(pos,
                                      z,
                                      c,
                                      tet_bxfx4[:, i *
                                                max_split:(i+1)*max_split, :],
                                      use_mask=False,
                                          cam_pos=cam_pos,
                                          cam_rot=cam_rot,
                                          cam_proj=cam_proj
                                          )

            pred_prob_list.append(pred.probs)
        if n_split*max_split < n_f:
            pred, _ = self.decode_occ(pos,
                                          z,
                                          c,
                                          tet_bxfx4[:, n_split * max_split:, :],
                                          use_mask=False,
                                          cam_pos=cam_pos,
                                          cam_rot=cam_rot,
                                          cam_proj=cam_proj
                                          )

            pred_prob_list.append(pred.probs)
        return torch.cat(pred_prob_list, dim=1)
