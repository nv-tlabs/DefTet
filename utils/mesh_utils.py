'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
import numpy as np
from layers.DefTet.tet_face_adj_m_idx.utils import tet_face_adj_m_f_idx
from layers.nearest_neighbor import NearestNeighbor
from layers.DefTet.tet_analytic_distance_batch.utils import tet_analytic_distance_f_batch

################################################################
def get_surface_normal_loss(vertices_bxnx3, faces_bxfx3):
    face = torch.gather(input=vertices_bxnx3.unsqueeze(dim=-2).expand(-1, -1, 3, -1),
                        index=faces_bxfx3.unsqueeze(dim=-1).expand(-1, -1, -1, 3),
                        dim=1)
    face_a = face[:, :, 0, :]
    face_b = face[:, :, 1, :]
    face_c = face[:, :, 2, :]

    normal_face = get_normal(face_a, face_b, face_c)
    # import ipdb
    # ipdb.set_trace()
    with torch.no_grad():
        one_face_adj_idx = tet_face_adj_m_f_idx(face[0].float())
    if one_face_adj_idx.sum() == 0:
        print('==> '
              'zero face shape')
        print(face.shape)
        return torch.zeros(vertices_bxnx3.shape[0], device=faces_bxfx3.device).float()

    normal_a = normal_face[:, one_face_adj_idx[0]]
    normal_b = normal_face[:, one_face_adj_idx[1]]
    normal_loss = 1 - torch.sum(normal_a * normal_b, dim=-1)

    return normal_loss.mean(dim=-1)


def get_normal(a, b, c):
    normal = torch.zeros_like(a)
    tmp_a = b - a
    tmp_b = c - a
    a = tmp_a
    b = tmp_b
    normal[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    normal[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    normal[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    normal = normal / \
             (torch.sqrt(torch.sum(normal ** 2, dim=-1, keepdim=True) + 1e-12))
    return normal


def sample_points_area_avg(vertices, faces, num_samples=10000, sample_uv=None):

    dist_uni = torch.distributions.Uniform(
        torch.tensor([0.0], device=vertices.device), torch.tensor([1.0], device=vertices.device
                                                                  ))

    x1, x2, x3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 0]) - torch.index_select(vertices, 0, faces[:, 1]), 1, dim=1)
    y1, y2, y3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 1]) - torch.index_select(vertices, 0, faces[:, 2]), 1, dim=1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    Areas = torch.sqrt(a + b + c) / 2
    # percentage of each face w.r.t. full surface area
    Areas = Areas / torch.sum(Areas)

    # define descrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(-1))
    face_choices = cat_dist.sample([num_samples])

    # from each face sample a point
    select_faces = faces[face_choices]
    xs = torch.index_select(vertices, 0, select_faces[:, 0])
    ys = torch.index_select(vertices, 0, select_faces[:, 1])
    zs = torch.index_select(vertices, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample([num_samples]))
    v = dist_uni.sample([num_samples])
    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
    if not sample_uv is None:
        uv_xs = torch.index_select(sample_uv, 0, select_faces[:, 0])
        uv_ys = torch.index_select(sample_uv, 0, select_faces[:, 1])
        uv_zs = torch.index_select(sample_uv, 0, select_faces[:, 2])
        pointsuv_ = (1 - u) * uv_xs + (u * (1 - v)) * uv_ys + u * v * uv_zs
        return points, face_choices, pointsuv_
    return points, face_choices

def sample_point_on_surface_fix_num(face_a, face_b, face_c, sample_all=100000, return_face_idx=False):
    # sample points in the surface.
    n_face = face_a.shape[0]

    each_face_num = int(sample_all / n_face) + 2


    u = torch.sqrt(torch.rand(
        size=(n_face, each_face_num, 1), device=face_a.device))
    v = torch.rand(size=(n_face, each_face_num, 1),
                   device=face_a.device)



    surface_p = (1 - u) * face_a.unsqueeze(1) + (u * (1 - v)) * \
                     face_b.unsqueeze(1) + u * v * face_c.unsqueeze(1)

    surface_p = surface_p.reshape(-1, 3)

    perm = torch.randperm(surface_p.shape[0], device=face_b.device)
    idx = perm[:sample_all]
    surface_p = surface_p[idx]
    if return_face_idx:
        face_idx = torch.arange(face_a.shape[0], device=face_a.device)
        face_idx = face_idx.unsqueeze(-1).expand(-1, each_face_num)
        face_idx = face_idx.reshape(-1)
        face_idx = face_idx[idx]
        return surface_p, face_idx
    return surface_p

def normalize_pc(p):
    # normalize such that the maximum box lengh is one
    l_max, _ = p.max(dim=0)
    l_min, _ = p.min(dim=0)
    l = l_max - l_min
    max_l = l.max()
    return p / max_l

def loadobj(meshfile, with_color=False):

    v = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if with_color:
            if len(data) != 4 or len(data) != 4 + 3:
                continue
            if data[0] == 'v':
                v.append([float(d) for d in data[1:4]])
        else:

            if len(data) != 4:
                continue
            if data[0] == 'v':
                v.append([float(d) for d  in data[1:]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3

def save_tetrahedron(point_px3, tetrahedron_fx4, f_name):
    with open(f_name, 'w') as f:
        all_str = ''
        for p in point_px3:
            all_str += 'v %f %f %f\n'%(p[0], p[1], p[2])

        for tetrahedron in tetrahedron_fx4:
            # we need to save 4 faces for this one
            tetrahedron = tetrahedron + 1
            all_str += 'f %d %d %d\n' % (tetrahedron[0], tetrahedron[1], tetrahedron[2])
            all_str += 'f %d %d %d\n' % (tetrahedron[0], tetrahedron[1], tetrahedron[3])
            all_str += 'f %d %d %d\n' % (tetrahedron[0], tetrahedron[2], tetrahedron[3])
            all_str += 'f %d %d %d\n' % (tetrahedron[3], tetrahedron[1], tetrahedron[2])
        f.write(all_str)


def cross_dot(a, b):
    normal = np.zeros(3)
    normal[0] = a[1] * b[2] - a[2] * b[1]
    normal[1] = a[2] * b[0] - a[0] * b[2]
    normal[2] = a[0] * b[1] - a[1] * b[0]
    return normal

def cross_dot_torch(a, b):
    normal = torch.zeros_like(a)
    normal[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    normal[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    normal[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return normal

def check_sign(a, b, c, d):
    normal = cross_dot(b - a, c - a)
    dot = normal * (d - a)
    dot = np.sum(dot)
    return dot

def check_tet(point_px3, tetrahedron_fx4):
    # for each face, the forth point should lie in the positive side of normal
    # 1, 2, 3, 4
    # 2, 1, 4, 3
    # 3, 4, 1, 2
    # 4, 3, 2, 1
    new_f = []
    dot_value = []
    for f in tetrahedron_fx4:
        dot_0 = check_sign(point_px3[f[0]], point_px3[f[1]], point_px3[f[2]], point_px3[f[3]])
        dot_1 = check_sign(point_px3[f[1]], point_px3[f[0]], point_px3[f[3]], point_px3[f[2]])
        dot_2 = check_sign(point_px3[f[2]], point_px3[f[3]], point_px3[f[0]], point_px3[f[1]])
        dot_3 = check_sign(point_px3[f[3]], point_px3[f[2]], point_px3[f[1]], point_px3[f[0]])
        # if dot_0 == dot_1 and dot_1 == dot_2 and dot_2 == dot_3:
        if dot_0 > 0 and dot_1 > 0 and dot_2 > 0 and dot_3 > 0:
            new_f.append(f)
            dot_value.append([dot_0, dot_1, dot_2, dot_3])
            continue
        import ipdb
        ipdb.set_trace()
        tmp_f = np.zeros_like(f)

    return new_f, dot_value

def save_tet_simple(tet_fx4x3, f_name):
    surface_idx = [[1, 2, 3],
                   [2, 1, 4],
                   [3, 4, 1],
                   [4, 3, 2]]
    surface_idx = np.asarray(surface_idx)
    with open(f_name, 'w') as f:
        all_str = ''
        for idx_tet, tetrahedron in enumerate(tet_fx4x3):
            # we need to save 4 faces for this one
            for i in range(4):
                all_str += 'v %f %f %f\n' % (tetrahedron[i][0], tetrahedron[i][1], tetrahedron[i][2])
            idx = idx_tet * 4
            for i in range(4):
                all_str += 'f %d %d %d\n' % (idx + surface_idx[i][0], idx + surface_idx[i][2], idx + surface_idx[i][1])
        f.write(all_str)

def tet_simple_to_verts(tet_fx4x3):
    surface_idx = [[1, 2, 3],
                   [2, 1, 4],
                   [3, 4, 1],
                   [4, 3, 2]]
    surface_idx = np.asarray(surface_idx)
    verts = []
    faces = []

    for idx_tet, tetrahedron in enumerate(tet_fx4x3):
        # we need to save 4 faces for this one
        for i in range(4):
            verts.append([tetrahedron[i][0], tetrahedron[i][1], tetrahedron[i][2]])

        idx = idx_tet * 4
        for i in range(4):
            faces.append([idx + surface_idx[i][0], idx + surface_idx[i][2], idx + surface_idx[i][1]])

    return verts, faces

def save_tet_face(tet_fx3x3, f_name):
    # use trimesh to fix the surface
    with open(f_name, 'w') as f:
        all_str = ''
        for idx_tri, triangle in enumerate(tet_fx3x3):
            for i in range(3):
                all_str += 'v %f %f %f\n' % (triangle[i][0], triangle[i][1], triangle[i][2])
            idx = idx_tri * 3
            all_str += 'f %d %d %d\n' % (idx + 1, idx + 3, idx + 2)
        f.write(all_str)

def tet_face_to_verts(tet_fx3x3):
    # use trimesh to fix the surface
    verts = []
    faces = []
    for idx_tri, triangle in enumerate(tet_fx3x3):
        for i in range(3):
           verts.append([triangle[i][0], triangle[i][1], triangle[i][2]])
        idx = idx_tri * 3
        faces.append([idx + 1, idx + 3, idx + 2])
    return np.asarray(verts), np.asarray(faces)

def sample_surf_point(face_fx3x3, each_face_num=20):
    a = face_fx3x3[:, 0:1, :]
    b = face_fx3x3[:, 1:2, :]
    c = face_fx3x3[:, 2:3, :]
    n_face = a.shape[0]
    u = torch.sqrt(torch.rand(size=(n_face, each_face_num, 1), device=face_fx3x3.device))
    v = torch.rand(size=(n_face, each_face_num, 1), device=face_fx3x3.device)
    specific_p = (1 - u) * a + (u * (1 - v)) * b + u * v * c
    return specific_p

def sample_surf_point_batch(face_bxfx3x3, each_face_num=20):
    a = face_bxfx3x3[:, :, 0:1, :]
    b = face_bxfx3x3[:, :, 1:2, :]
    c = face_bxfx3x3[:, :, 2:3, :]
    n_face = a.shape[1]
    n_batch = a.shape[0]
    u = torch.sqrt(torch.rand(size=(n_batch, n_face, each_face_num, 1), device=face_bxfx3x3.device))
    v = torch.rand(size=(n_batch, n_face, each_face_num, 1), device=face_bxfx3x3.device)
    specific_p = (1 - u) * a + (u * (1 - v)) * b + u * v * c
    return specific_p

def save_mesh_color(pointnp_px3, points_color_px3, facenp_fx3, fname, partinfo=None):

    fid = open(fname, 'w')
    ss = ''
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        ss += 'v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], points_color_px3[pidx][0], points_color_px3[pidx][1], points_color_px3[pidx][2])
    for f in facenp_fx3:
        f1 = f + 1
        ss += 'f %d %d %d\n' % (f1[0], f1[1], f1[2])
    fid.write(ss)
    fid.close()
    # else:
    #     fid = open(fname, 'w')
    #     for pidx, p in enumerate(pointnp_px3):
    #         if partinfo[pidx, -1] == 0:
    #             pp = p
    #             color = [1, 0, 0]
    #         else:
    #             pp = p
    #             color = [0, 0, 1]
    #         fid.write('v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
    #     for f in facenp_fx3:
    #         f1 = f + 1
    #         fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
    #     fid.close()
    return

def save_mesh(pointnp_px3, facenp_fx3, fname, partinfo=None):
    if partinfo is None:
        fid = open(fname, 'w')
        ss = ''
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            ss += 'v %f %f %f\n' % (pp[0], pp[1], pp[2])
        for f in facenp_fx3:
            f1 = f + 1
            ss += 'f %d %d %d\n' % (f1[0], f1[1], f1[2])
        fid.write(ss)
        fid.close()
    else:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            if partinfo[pidx, -1] == 0:
                pp = p
                color = [1, 0, 0]
            else:
                pp = p
                color = [0, 0, 1]
            fid.write('v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    return


EPS = 1e-10

def point_point_distance(a_bxnx3, b_bxmx3):
    side_distance = NearestNeighbor()
    closest_index_in_S2 = side_distance(a_bxnx3, b_bxmx3)
    closest_S2 = torch.gather(input=b_bxmx3, dim=1, index=closest_index_in_S2.unsqueeze(-1).expand(-1, -1, 3))
    chamfer_distance = torch.sqrt(
        torch.sum((a_bxnx3 - closest_S2) ** 2, dim=-1) + EPS)
    return chamfer_distance

def point_mesh_distance(a_bxnx3, mesh_bxfx3):
    batch_surface_length = torch.zeros(mesh_bxfx3.shape[0], device=mesh_bxfx3.device).float()
    batch_surface_length += mesh_bxfx3.shape[1]
    tet_distance, _ = tet_analytic_distance_f_batch(a_bxnx3, mesh_bxfx3, batch_surface_length)

    tet_distance = torch.sqrt(tet_distance + EPS)
    return tet_distance
