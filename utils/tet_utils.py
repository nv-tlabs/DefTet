'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import numpy as np
import torch
from tqdm import tqdm
# import pymesh
from collections import defaultdict
from scipy.sparse import coo_matrix
from utils.lib.tet_point_adj.interface import Tet_point_adj
from utils.lib.tet_face_adj.interface import Tet_face_adj
from utils.lib.tet_adj_share.interface import Tet_adj_share
from utils.matrix_utils import convert_torch_sparse, sparse_batch_matmul

c_tet_point_adj = Tet_point_adj()
c_tet_face_adj = Tet_face_adj()
c_obj_tet_adj_share = Tet_adj_share()


def scaler_triplet_produt(a, b, c):
    return torch.sum(a * torch.cross(b, c), dim=-1)

def bary_centric_tet(a, b, c, d, p):
    vap = p - a
    vbp = p - b

    vab = b - a
    vac = c - a
    vad = d - a

    vbc = c - b
    vbd = d - b

    va6 = scaler_triplet_produt(vbp, vbd, vbc)
    vb6 = scaler_triplet_produt(vap, vac, vad)
    vc6 = scaler_triplet_produt(vap, vad, vab)
    vd6 = scaler_triplet_produt(vap, vab, vac)
    v6 = 1 / scaler_triplet_produt(vab, vac, vad)

    return va6*v6, vb6*v6, vc6*v6, vd6*v6

def tet_to_adj_sparse(points, tet_list, normalize=False):
    # use sparse matrix here to better memory usage
    n_point = points.shape[0]
    adj_list = set()
    for tet in tqdm(tet_list):
        adj_list.add((tet[0], tet[1]))
        adj_list.add((tet[0], tet[2]))
        adj_list.add((tet[0], tet[3]))

        adj_list.add((tet[1], tet[0]))
        adj_list.add((tet[1], tet[2]))
        adj_list.add((tet[1], tet[3]))

        adj_list.add((tet[2], tet[0]))
        adj_list.add((tet[2], tet[1]))
        adj_list.add((tet[2], tet[3]))

        adj_list.add((tet[3], tet[0]))
        adj_list.add((tet[3], tet[1]))
        adj_list.add((tet[3], tet[2]))

    new_adj_list = []
    for a, b in adj_list:
        new_adj_list.append([int(a), int(b)])
    idx = np.asarray(new_adj_list)
    v = np.ones(idx.shape[0])
    if normalize:
        adj_m = coo_matrix((v, (idx[:, 0], idx[:, 1])),
                         shape=(n_point, n_point))

        sum_adj = 1.0 / adj_m.sum(axis=-1)
        n_point = sum_adj.shape[0]
        new_idx = list(range(n_point))
        sum_m = coo_matrix((np.asarray(sum_adj).reshape(-1),
                            (new_idx, new_idx)), shape=(n_point, n_point))
        adj = sum_m.dot(adj_m)
        idx = np.asarray(adj.nonzero())
        adj = torch.sparse.FloatTensor(torch.from_numpy(idx).long(
        ), torch.from_numpy(adj.data).float(), torch.Size([n_point, n_point]))
    else:
        idx = torch.from_numpy(idx)
        v = torch.ones(idx.shape[0])
        adj = torch.sparse.FloatTensor(idx.transpose(
            0, 1), v, torch.Size([n_point, n_point]))

    return adj

def c_tet_to_adj_sparse(points, tet_list, normalize=True):
    return c_tet_point_adj.run(points.shape[0], tet_list.astype(np.int32), normalize)

def tet_to_adj( points, tet_list):
    n_point = points.shape[0]
    adj = np.zeros((n_point, n_point))
    for tet in tqdm(tet_list):
        adj[tet[0]][tet[1]] = 1
        adj[tet[0]][tet[2]] = 1
        adj[tet[0]][tet[3]] = 1

        adj[tet[1]][tet[0]] = 1
        adj[tet[1]][tet[2]] = 1
        adj[tet[1]][tet[3]] = 1

        adj[tet[2]][tet[1]] = 1
        adj[tet[2]][tet[0]] = 1
        adj[tet[2]][tet[3]] = 1

        adj[tet[3]][tet[1]] = 1
        adj[tet[3]][tet[0]] = 1
        adj[tet[3]][tet[2]] = 1
    adj = torch.from_numpy(adj)
    return adj

def tet_to_adj_with_self_sparse(points, tet_list):
    n_point = points.shape[0]
    adj_list = set()
    for tet in tqdm(tet_list):
        # Self
        adj_list.add((tet[0], tet[0]))
        adj_list.add((tet[1], tet[1]))
        adj_list.add((tet[2], tet[2]))
        adj_list.add((tet[3], tet[3]))

        adj_list.add((tet[0], tet[1]))
        adj_list.add((tet[0], tet[2]))
        adj_list.add((tet[0], tet[3]))

        adj_list.add((tet[1], tet[0]))
        adj_list.add((tet[1], tet[2]))
        adj_list.add((tet[1], tet[3]))

        adj_list.add((tet[2], tet[0]))
        adj_list.add((tet[2], tet[1]))
        adj_list.add((tet[2], tet[3]))

        adj_list.add((tet[3], tet[0]))
        adj_list.add((tet[3], tet[1]))
        adj_list.add((tet[3], tet[2]))

    new_adj_list = []
    for a, b in adj_list:
        new_adj_list.append([int(a), int(b)])

    idx = torch.from_numpy(np.asarray(new_adj_list))
    v = torch.ones(idx.shape[0])
    adj = torch.sparse.FloatTensor(idx.transpose(
        0, 1), v, torch.Size([n_point, n_point]))
    return adj

def tet_to_face_adj_sparse(points, tet_list):
    # use sparse matrix here to better memory usage
    n_point = points.shape[0]
    edge_idx = defaultdict(list)
    absolute_face_idx = dict()
    idx_array = [0, 1, 2,
                 1, 0, 3,
                 2, 3, 0,
                 3, 2, 1]
    idx_array = np.asarray(idx_array).reshape(4, 3)
    for tet_idx, tet in tqdm(enumerate(tet_list)):
        idx_list = [[tet[idx[0]], tet[idx[1]], tet[idx[2]]]
                    for idx in idx_array]
        # four face index ot this place
        face_idx = [tet_idx * 4 + i for i in range(4)]
        # adj_face: using edge as the adj

        for i_face, triangle in enumerate(idx_list):
            for i_edge in range(3):
                point_a = min(triangle[i_edge], triangle[(i_edge + 1) % 3])
                point_b = max(triangle[i_edge], triangle[(i_edge + 1) % 3])
                edge_idx[point_a * n_point +
                         point_b].append(face_idx[i_face])
            face_p_a = min(idx_list[i_face])
            face_p_b = max(idx_list[i_face])
            for p in idx_list[i_face]:
                if p != face_p_a and p != face_p_b:
                    face_p_c = p
            absolute_face_idx[face_idx[i_face]] = face_p_a * \
                (n_point ** 2) + face_p_b * n_point + face_p_c

    n_face = len(tet_list) * 4
    new_adj_list = []
    for edge in tqdm(edge_idx.keys()):
        for face_a in edge_idx[edge]:
            for face_b in edge_idx[edge]:
                if face_a == face_b:
                    continue
                if absolute_face_idx[face_a] == absolute_face_idx[face_b]:
                    continue
                new_adj_list.append([face_a, face_b])

    idx = np.asarray(new_adj_list)
    v = np.ones(idx.shape[0])
    adj = coo_matrix((v, (idx[:, 0], idx[:, 1])),
                     shape=(n_face, n_face))
    return adj

def c_tet_to_face_adj_sparse(points, tet_list):
    # use sparse matrix here to better memory usage
    return c_tet_face_adj.run(points.shape[0], tet_list.astype(np.int32))


def tet_to_face(n_point, tet_list):
    # use sparse matrix here to better memory usage
    tet_face_fx3 = []
    tet_face_tetidx_fx2 = []
    tet_face_tetfaceidx_fx2 = []

    absolute_face_idx = dict()

    idx_array = [0, 1, 2,
                 1, 0, 3,
                 2, 3, 0,
                 3, 2, 1]

    idx_array = np.asarray(idx_array).reshape(4, 3)
    for tet_idx, tet in tqdm(enumerate(tet_list)):
        idx_list = [[tet[idx[0]], tet[idx[1]], tet[idx[2]]]
                    for idx in idx_array]
        for i_face, triangle in enumerate(idx_list):
            face_p_a = min(idx_list[i_face])
            face_p_b = max(idx_list[i_face])
            for p in idx_list[i_face]:
                if p != face_p_a and p != face_p_b:
                    face_p_c = p
            face_idx = face_p_a * (n_point ** 2) + \
                face_p_b * n_point + face_p_c
            if face_idx not in absolute_face_idx:
                absolute_face_idx[face_idx] = [
                    [triangle], [tet_idx], [i_face]]
            else:
                absolute_face_idx[face_idx][0].append(triangle)
                absolute_face_idx[face_idx][1].append(tet_idx)
                absolute_face_idx[face_idx][2].append(i_face)

    # only consider the face that has two tet idx
    cnt_n_tet = [0, 0, 0]
    tet_boundary_face = []
    for face_idx in absolute_face_idx.keys():
        if len(absolute_face_idx[face_idx][0]) == 2:
            cnt_n_tet[1] += 1
            tet_face_fx3.append(absolute_face_idx[face_idx][0][0])
            tet_face_tetidx_fx2.append(absolute_face_idx[face_idx][1])
            tet_face_tetfaceidx_fx2.append(absolute_face_idx[face_idx][2])
        elif len(absolute_face_idx[face_idx][0]) == 1:
            cnt_n_tet[0] += 1
            tet_boundary_face.append(absolute_face_idx[face_idx][0][0])
        else:
            cnt_n_tet[2] += 1
    print('Cnt neighbor tet: ', cnt_n_tet)
    return np.asarray(tet_face_fx3), np.asarray(tet_face_tetidx_fx2), np.asarray(tet_face_tetfaceidx_fx2), np.asarray(tet_boundary_face)


def tet_to_face_withtet(points, tet_list):
    # use sparse matrix here to better memory usage
    tet_face_tetidx_fx2 = []

    n_point = points.shape[0]
    absolute_face_idx = dict()
    absolute_face_to_tet = defaultdict(list)

    idx_array = [0, 1, 2,
                 1, 0, 3,
                 2, 3, 0,
                 3, 2, 1]

    idx_array = np.asarray(idx_array).reshape(4, 3)
    for tet_idx, tet in tqdm(enumerate(tet_list)):
        idx_list = [[tet[idx[0]], tet[idx[1]], tet[idx[2]]]
                    for idx in idx_array]
        # four face index ot this place
        face_idx = [tet_idx * 4 + i for i in range(4)]
        # adj_face: using edge as the adj

        for i_face, triangle in enumerate(idx_list):
            face_p_a = min(idx_list[i_face])
            face_p_b = max(idx_list[i_face])
            for p in idx_list[i_face]:
                if p != face_p_a and p != face_p_b:
                    face_p_c = p
            t = face_p_a * (n_point ** 2) + face_p_b * n_point + face_p_c
            absolute_face_idx[face_idx[i_face]] = t
            absolute_face_to_tet[t].append(tet_idx)

    for i in range(len(tet_list) * 4):
        tet = absolute_face_to_tet[absolute_face_idx[i]]
        if len(tet) == 1:
            tet.append(0)
        elif len(tet) == 0:
            raise ValueError
        elif len(tet) > 2:
            raise ValueError
        tet_face_tetidx_fx2.append(tet)

    return np.asarray(tet_face_tetidx_fx2)

def min_3(a, b, c):
    r = a
    if (b < r):
        r = b
    if (c < r):
        r = c
    return r

def max_3(a, b, c):
    r = a
    if (b > r):
        r = b
    if (c > r):
        r = c
    return r

def tet_adj_share(tet_list, n_point):
    n_tet = tet_list.shape[0]
    # using sparse matric
    index_list = []
    idx_array = [0, 1, 2,
                 1, 0, 3,
                 2, 3, 0,
                 3, 2, 1]
    idx_array = np.asarray(idx_array).reshape(4, 3)
    ########################################################
    # A naive implementation version.  You need to check cuda again!!!! some problem will happen for this!!!!
    face_index = dict()
    for tet_idx, tet in enumerate(tet_list):
        for i_face in range(4):
            a = min_3(tet[idx_array[i_face][0]], tet[idx_array[i_face][1]], tet[idx_array[i_face][2]])
            b = max_3(tet[idx_array[i_face][0]], tet[idx_array[i_face][1]], tet[idx_array[i_face][2]])
            for i in range(3):
                if a != tet[idx_array[i_face][i]] and b != tet[idx_array[i_face][i]]:
                    c = tet[idx_array[i_face][i]]
            num = a * n_point * n_point + b * n_point + c

            if (num not in face_index) :
                face_index[num] = [[tet_idx, i_face]]
            else:
                face_index[num].append([tet_idx, i_face])


    singular_face = 0
    # check the face index for each tet
    for face_num in face_index.keys():
        all_tet = face_index[face_num]
        if len(all_tet) == 1:
            singular_face += 1
        if len(all_tet) == 2:
            index_list.append(
                [all_tet[0][0], all_tet[1][0], all_tet[0][1]])
            index_list.append(
                [all_tet[1][0], all_tet[0][0], all_tet[1][1]])
        if len(all_tet) > 2:
            raise ValueError
    index_list = np.asarray(index_list)
    index_value = np.ones(index_list.shape[0])

    adj_list = []
    for i in range(4):
        adj_list.append(coo_matrix((index_value[index_list[:, 2] == i],
                                    (index_list[:, 0][index_list[:, 2] == i], index_list[:, 1][index_list[:, 2] == i])),
                                   shape=(n_tet, n_tet)))
    adj_list = [convert_torch_sparse(adj) for adj in adj_list]
    return adj_list



def c_tet_adj_share(tet_list, n_point, torch_t=True):
    adj_list = c_obj_tet_adj_share.run(tet_list.astype(np.int32), n_point)
    if torch_t:
        adj_list = [convert_torch_sparse(adj) for adj in adj_list]
    return adj_list


def read_tet(file_name):
    tetrahedrons = []
    vertices = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = line.strip().split(' ')
        n_vert = int(line[1])
        n_t = int(line[2])
        for i in range(n_vert):
            line = f.readline()
            line = line.strip().split(' ')
            assert len(line) == 3
            vertices.append([float(v) for v in line])
        for i in range(n_t):
            line = f.readline()
            line = line.strip().split(' ')
            assert len(line) == 4
            tetrahedrons.append([int(v) for v in line])

    assert len(tetrahedrons) == n_t
    assert len(vertices) == n_vert
    vertices = np.asarray(vertices)
    return vertices, np.asarray(tetrahedrons)

def save_tetrahedron(point_px3, tetrahedron_fx4, f_name):
    with open(f_name, 'w') as f:
        for p in point_px3:
            f.write('v %f %f %f\n' % (p[0], p[1], p[2]))

        for tetrahedron in tetrahedron_fx4:
            # we need to save 4 faces for this one
            tetrahedron = tetrahedron + 1
            f.write('f %d %d %d\n' %
                    (tetrahedron[0], tetrahedron[1], tetrahedron[2]))
            f.write('f %d %d %d\n' %
                    (tetrahedron[0], tetrahedron[1], tetrahedron[3]))
            f.write('f %d %d %d\n' %
                    (tetrahedron[0], tetrahedron[2], tetrahedron[3]))
            f.write('f %d %d %d\n' %
                    (tetrahedron[3], tetrahedron[1], tetrahedron[2]))


def get_tet_adj(tetrahedron_fx4,  n_point):
    tet_face_adj_list = tet_adj_share(
        tetrahedron_fx4, n_point)  # checked correct :)
    tet_face_adj_list = [adj.cuda() for adj in tet_face_adj_list]
    return tet_face_adj_list


def get_face_use_occ(tet_bxfx4x3, center_occ_cuda, tet_adj):
    # tet_bxfx4x3: bxtx4x3
    # center_occ_cuda: bxtx1
    # tet_adj: get it by function  get_tet_adj()
    batch_size = tet_bxfx4x3.shape[0]
    all_equal_list = []

    for i in range(4):
        neibor_occ = sparse_batch_matmul(
            tet_adj[i].float(), center_occ_cuda.float())
        # have different neighbor
        equal = (neibor_occ != center_occ_cuda)

        equal = equal & (center_occ_cuda == 1)
        has_adj = torch.sparse.sum(tet_adj[i], dim=1).to_dense().bool().unsqueeze(0).unsqueeze(-1)
        equal = equal & has_adj
        all_equal_list.append(equal)
    all_equal_list = torch.cat(all_equal_list, dim=-1)


    A = tet_bxfx4x3[:, :, 0:1, :]
    B = tet_bxfx4x3[:, :, 1:2, :]
    C = tet_bxfx4x3[:, :, 2:3, :]
    D = tet_bxfx4x3[:, :, 3:4, :]

    all_face_a = torch.cat([A, B, C, D], dim=2)
    all_face_b = torch.cat([B, A, D, C], dim=2)
    all_face_c = torch.cat([C, D, A, B], dim=2)
    all_equal_list = all_equal_list.unsqueeze(
        -1).expand_as(all_face_a).byte()

    # all_equal_list_np = all_equal_list.data.cpu().numpy()
    all_face = []
    for i_batch in range(batch_size):
        equal = all_equal_list[i_batch]
        face_a = all_face_a[i_batch][equal]
        face_a = face_a.reshape(-1, 1, 3)
        face_b = all_face_b[i_batch][equal]
        face_b = face_b.reshape(-1, 1, 3)
        face_c = all_face_c[i_batch][equal]
        face_c = face_c.reshape(-1, 1, 3)
        face = torch.cat([face_a, face_b, face_c], dim=1)
        all_face.append(face)
        # face: fx3x3
    return all_face

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

def save_tet(pos, tet_list, f_name):
    with open(f_name, 'w') as f:
        all_str = ''
        for p in pos:
            all_str += 'v %f %f %f\n' % (p[0], p[1], p[2])
        for t in tet_list:
            all_str += 't %d %d %d %d\n'%(t[0], t[1], t[2], t[3])

        f.write(all_str)


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