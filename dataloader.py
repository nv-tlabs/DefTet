'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import kaolin as kal
import torch
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from utils.mesh_utils import save_mesh
class MakeSurfaceMesh:
    def __init__(self, resolution=100, smoothing_iterations=3, save_preprocess=False, max_length=0.9):
        self.resolution = resolution
        self.smoothing_iterations = smoothing_iterations
        self.save_preprocess = save_preprocess
        self.max_length = max_length
        self.error_idx = []

    def __call__(self, mesh):
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()
        max_l = max(vertices[..., 0].max() - vertices[..., 0].min(),
                    vertices[..., 1].max() - vertices[..., 1].min(),
                    vertices[..., 2].max() - vertices[..., 2].min())
        vertices = (vertices / max_l) * self.max_length
        mid_p = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
        vertices = vertices - mid_p.unsqueeze(dim=0)
        voxelgrid = kal.ops.conversions.trianglemeshes_to_voxelgrids(
                vertices.unsqueeze(0), faces,
                resolution=self.resolution)

        odms = kal.ops.voxelgrid.extract_odms(voxelgrid)
        voxelgrid = kal.ops.voxelgrid.project_odms(odms)
        # convert back to voxelgrids
        new_vertices, new_faces = kal.ops.conversions.voxelgrids_to_trianglemeshes(
            voxelgrid,
        )
        new_vertices = new_vertices[0]
        new_faces = new_faces[0]
        # laplacian smoothing
        adj_mat = kal.ops.mesh.adjacency_matrix(
            new_vertices.shape[0],
            new_faces)
        num_neighbors = torch.sparse.sum(
            adj_mat, dim=1).to_dense().view(-1, 1)
        for i in range(self.smoothing_iterations):
            neighbor_sum = torch.sparse.mm(adj_mat, new_vertices)
            new_vertices = neighbor_sum / num_neighbors
        # normalize
        orig_min = vertices.min(dim=0)[0]
        orig_max = vertices.max(dim=0)[0]
        new_min = new_vertices.min(dim=0)[0]
        new_max = new_vertices.max(dim=0)[0]
        new_vertices = (new_vertices - new_min) / (new_max - new_min)
        new_vertices = new_vertices * (orig_max - orig_min) + orig_min
        return new_vertices.cpu(), new_faces.cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'watertight_%s'%(str(datetime.now()))
        return 'watertight'

class SamplePointsFromMesh:
    def __init__(self, num_points, with_normals=True, save_preprocess=False):
        self.num_points = num_points
        self.with_normals = with_normals
        self.save_preprocess = save_preprocess
    def __call__(self, mesh):
        vertices = mesh[0].unsqueeze(dim=0).float().cuda()
        faces = mesh[1].long().cuda()
        points, face_choices = kal.ops.mesh.sample_points(
            vertices, faces, self.num_points)
        if self.with_normals:
            face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)
            face_normals = kal.ops.mesh.face_normals(
                face_vertices, unit=True)
            normals = face_normals[face_choices]
            return points.squeeze(0), normals.squeeze(0)
        return points.squeeze(0).cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'point_cloud_%s'%(str(datetime.now()))
        return 'point_cloud'

def kaolin_mesh_to_sdf(verts_bxnx3, face_fx3, points_bxnx3):
    sign = kal.ops.mesh.check_sign(verts_bxnx3, face_fx3, points_bxnx3, hash_resolution=512)
    face_vertices = kal.ops.mesh.index_vertices_by_faces(verts_bxnx3, face_fx3)
    distance, index, dist_type = kal.metrics.trianglemesh.point_to_mesh_distance(points_bxnx3, face_vertices)
    sign = sign.float() * 2.0 - 1.0  # (1: inside; -1: outside)
    sdf = sign * distance
    return sdf


class SDFPoints:
    def __init__(self, num_points, with_normals=True, save_preprocess=False):
        self.num_points = num_points
        self.with_normals = with_normals
        self.save_preprocess = save_preprocess
    def __call__(self, mesh):
        vertices = mesh[0].unsqueeze(dim=0).float().cuda()
        faces = mesh[1].long().cuda()
        points = 1.05 * (torch.rand(1, self.num_points, 3).cuda() - .5)
        sdf = kaolin_mesh_to_sdf(vertices, faces, points)
        return points[0].cpu(), sdf[0].cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'sdf_%s'%(str(datetime.now()))
        return 'sdf'


def create_dataloader(shapenet_source='/data/shapenet_kaolin/ShapeNet/objects',
                      save_cache_root = '/root/shapenet/',
                      train=True, batch_size=8, add_occupancy=False, only_chairs=False):

    train_cat = [   '02691156',
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
    if only_chairs:
        train_cat = [
            '03001627',
        ]

    # train_cat = ['02958343'] # car shape##########
    ds = kal.io.shapenet.ShapeNetV1(root=shapenet_source, categories=train_cat,
                                    with_materials=False, train=train)

    error_model = ['04090263_4a32519f44dc84aabafe26e2eb69ebf4'] # This one has no mesh :(
    error_idx = [ds.names.index(e) for e in error_model if e in ds.names]
    for idx in error_idx:
        ds.paths.pop(idx)
        ds.synset_idxs.pop(idx)
        ds.names.pop(idx)

    sv_dir = os.path.join(save_cache_root, 'watertight')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    print('==> preprocess watertight mesh')

    watertight_mesh = kal.io.dataset.ProcessedDataset(
        ds, MakeSurfaceMesh(100, 3, save_preprocess=True), num_workers=0,
        cache_dir=sv_dir)


    sv_dir = os.path.join(save_cache_root, 'pcd')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

    print('==> preprocess point cloud')
    ####################
    processed_ds = kal.io.dataset.ProcessedDataset(
        watertight_mesh, SamplePointsFromMesh(100000, with_normals=False, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)

    print('==> preprocess sdf')
    sv_dir = os.path.join(save_cache_root, 'sdf')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    occ_dataset = kal.io.dataset.ProcessedDataset(
        watertight_mesh, SDFPoints(100000, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)
    #########


    combined_dataset = kal.io.dataset.CombinationDataset([watertight_mesh, processed_ds,
                                                          occ_dataset])

    def collate_fn(batch_list):
        data = dict()

        data['verts'] = [da[0][0][0] for da in batch_list]
        data['faces'] = [da[0][0][1] for da in batch_list]
        data['sample_points'] = torch.cat([da[0][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['name'] = [da[1][0]['name'] for da in batch_list]
        data['synset'] = [da[1][0]['synset'] for da in batch_list]
        data['sdf_point'] = torch.cat([da[0][2][0].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['sdf_value'] = torch.cat([da[0][2][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        return data

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        drop_last=True,
    )##### We always shuffle the data here
    return dataloader

if __name__ == '__main__':
    # dataloader = create_dataloader(train=False, only_chairs=False)
    dataloader = create_dataloader(train=False, only_chairs=False)
    print('==> finished validatation data')
    # dataloader_val = create_dataloader(train=False, only_chairs=True)
    save_folder = '/root/shapenet_car_all_update'
    os.makedirs(save_folder, exist_ok=True)
    from tqdm import tqdm
    cnt = 0
    for data in tqdm(iter(dataloader)):
        # import ipdb
        # ipdb.set_trace()######
        mesh_v_list = data['verts']
        mesh_f_list = data['faces']
        name_list = data['name']
        for v, f, n in zip(mesh_v_list, mesh_f_list, name_list):
            # import ipdb
            # ipdb.set_trace()
            save_mesh(v.data.cpu().numpy(), f.data.cpu().numpy(), os.path.join(save_folder, n  + '.obj'))
            cnt += 1
            if cnt > 100:
                exit()




