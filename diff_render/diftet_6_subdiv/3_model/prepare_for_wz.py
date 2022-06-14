'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from tqdm import tqdm


############################################
def read_tetrahedron(file_name, res=0.02):

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
    vertices = np.asarray(vertices, dtype=np.float32)
    vertices[vertices <= (0 + res / 4.0)] = 0  # determine the boundary point
    vertices[vertices >= (1 - res / 4.0)] = 1  # determine the boundary point
    mask = np.logical_and(vertices < 1, vertices > 0)

    return vertices, np.asarray(tetrahedrons, dtype=np.int64), mask


######################################################################
def tet_to_face_idx(n_point, tet_list, with_boundary=False):

    tet_face_fx3 = []
    tet_face_tetidx_fx2 = []
    tet_face_tetfaceidx_fx2 = []

    # n_point = points.shape[0]
    absolute_face_idx = dict()

    idx_array = [0, 1, 2, 1, 0, 3, 2, 3, 0, 3, 2, 1]

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
                absolute_face_idx[face_idx] = [[triangle], [tet_idx], [i_face]]
            else:
                absolute_face_idx[face_idx][0].append(triangle)
                absolute_face_idx[face_idx][1].append(tet_idx)
                absolute_face_idx[face_idx][2].append(i_face)

    # only consider the face that has two tet idx
    cnt_n_tet = [0, 0, 0]
    for face_idx in absolute_face_idx.keys():
        if len(absolute_face_idx[face_idx][0]) == 2:
            cnt_n_tet[1] += 1
            tet_face_fx3.append(absolute_face_idx[face_idx][0][0])
            tet_face_tetidx_fx2.append(absolute_face_idx[face_idx][1])
            tet_face_tetfaceidx_fx2.append(absolute_face_idx[face_idx][2])
        elif len(absolute_face_idx[face_idx][0]) == 1:
            if with_boundary:
                tet_face_fx3.append(absolute_face_idx[face_idx][0][0])
                tet_face_tetidx_fx2.append(
                    [absolute_face_idx[face_idx][1][0], -1])
                tet_face_tetfaceidx_fx2.append(
                    [absolute_face_idx[face_idx][2][0], -1])
            cnt_n_tet[0] += 1
        else:
            cnt_n_tet[2] += 1
    print('Cnt neighbor tet: ', cnt_n_tet)

    return np.asarray(tet_face_fx3,
                      dtype=np.int64), np.asarray(tet_face_tetidx_fx2,
                                                  dtype=np.int64), np.asarray(
                                                      tet_face_tetfaceidx_fx2,
                                                      dtype=np.int64)


##########################################################
def generate_point_adj(n_point, tet_list):

    # n_point = points.shape[0]
    adj = np.zeros((n_point, n_point), dtype=np.float32)
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

    return adj


def generate_point_adj_idx(n_point, tet_list):

    # point adj mtx
    # it should be a sparse matrix
    # fix it later
    tetpointadj_pxp = generate_point_adj(n_point, tet_list)

    # convert it to index
    adjsum = np.sum(tetpointadj_pxp, axis=1)
    adjmax = adjsum.max()
    pointadj_idx_pxm = -np.ones(shape=(n_point, int(adjmax)), dtype=np.int64)
    for i, adj_p in tqdm(enumerate(tetpointadj_pxp)):
        nozero = np.where(adj_p > 0)[0]
        pointadj_idx_pxm[i, :int(adjsum[i])] = nozero
    return pointadj_idx_pxm, adjsum.reshape(-1, 1)


def generate_tet2point_adj(points_px3, tet_list_tx4):
    '''
    convert tet properties to points properties
    point_property_pxd = mtx_pxt * tet_property_txd
    '''

    n_point = points_px3.shape[0]
    n_tet = tet_list_tx4.shape[0]

    mtx_pxt = np.zeros((n_point, n_tet), dtype=np.float32)
    adj = mtx_pxt

    for i, tet in tqdm(enumerate(tet_list_tx4)):
        adj[tet[0], i] = 1
        adj[tet[1], i] = 1
        adj[tet[2], i] = 1
        adj[tet[3], i] = 1

    return mtx_pxt


##########################################################################
def delete_tet(tet_list_tx4, tet_weights_tx4, thres=0.01):
    '''
    delete tets which has too small weights
    here, we just delete face but keep points
    '''
    tet_weights_t = np.max(tet_weights_tx4, axis=1)
    kept_tet = tet_weights_t > thres
    tet_list_kx4 = tet_list_tx4[kept_tet]

    return tet_list_kx4


##################################################################
def generate_edge(tet_list_tx4):
    r"""
    Agrs:
        tet_list_tx4: tet index
    """
    edges_all = []
    edges_all_connect_6x2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for edge_connect in edges_all_connect_6x2:
        idxbe, idxen = edge_connect
        edgebe = tet_list_tx4[:, idxbe]
        edgeen = tet_list_tx4[:, idxen]
        edgebe, edgeen = np.minimum(edgebe, edgeen), np.maximum(edgebe, edgeen)
        edge_tx2 = np.stack([edgebe, edgeen], axis=1)
        edges_all.append(edge_tx2)

    edges_all_ex2 = np.concatenate(edges_all, axis=0)
    # get rid of redunct
    edges_all_ex2 = np.unique(edges_all_ex2, axis=0)

    return edges_all_ex2


def matchedgelist(edgebe_k, edgeen_k, edge_list_ex2):
    '''
    match1 = edgebe_k.reshape(-1, 1) - edge_list_ex2[:, 0].reshape(1, -1)
    match2 = edgeen_k.reshape(-1, 1) - edge_list_ex2[:, 1].reshape(1, -1)
    match = np.abs(match1) + np.abs(match2)
    edgedix = np.where(match == 0)[1]
    '''

    edgedix = []
    edge_all_query = np.stack([edgebe_k, edgeen_k], axis=1)
    for edge_query in tqdm(edge_all_query):
        edge_match = edge_query == edge_list_ex2
        idx = np.where(edge_match[:, 0] & edge_match[:, 1])[0][0]
        edgedix.append(idx)

    return np.asarray(edgedix)


def generate_tet_edge_idx(tet_list_tx4, edges_all_ex2):

    tet_edge_list_tx6 = []
    edges_all_connect_6x2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for edge_connect in edges_all_connect_6x2:
        idxbe, idxen = edge_connect
        edgebe = tet_list_tx4[:, idxbe]
        edgeen = tet_list_tx4[:, idxen]
        edgebe, edgeen = np.minimum(edgebe, edgeen), np.maximum(edgebe, edgeen)
        edgedix = matchedgelist(edgebe, edgeen, edges_all_ex2)
        tet_edge_list_tx6.append(edgedix)

    return np.stack(tet_edge_list_tx6, axis=1)


def generate_edge_points(tet_points_px3, tet_feat_pxk, edges_all_ex2):

    # now we use middle point
    # in the future we may use advanced methods

    pbe = tet_points_px3[edges_all_ex2[:, 0], :]
    pen = tet_points_px3[edges_all_ex2[:, 1], :]
    edge_middle_points = (pbe + pen) / 2

    pfeatbe = tet_feat_pxk[edges_all_ex2[:, 0], :]
    pfeaten = tet_feat_pxk[edges_all_ex2[:, 1], :]
    edge_middle_feat = (pfeatbe + pfeaten) / 2

    return edge_middle_points, edge_middle_feat


def generate_subdivision(tet_list_tx4,
                         tet_points_px3,
                         tet_feat_pxk,
                         tet_list_subdiv_sig=None):

    # 1) get edge
    edges_all_ex2 = generate_edge(tet_list_tx4)

    # 2) relate tet to edge
    tet_list_edge_idx_tx6 = generate_tet_edge_idx(tet_list_tx4, edges_all_ex2)

    # 3) for each edge, we insert a new vertex
    edge_middle_points, edge_middle_feat = generate_edge_points(
        tet_points_px3, tet_feat_pxk, edges_all_ex2)

    # 4) now, we have vertex and newly created edge middle points
    tet_points_new_Px3 = np.concatenate([tet_points_px3, edge_middle_points],
                                        axis=0)
    tet_feat_new_Pxk = np.concatenate([tet_feat_pxk, edge_middle_feat], axis=0)

    # 5) now we delete old tet, create new tet
    pnum = tet_points_px3.shape[0]
    idx_a, idx_b, idx_c, idx_d = tet_list_tx4.T
    idx_ab, idx_ac, idx_ad, idx_bc, idx_bd, idx_cd = (tet_list_edge_idx_tx6 +
                                                      pnum).T

    tet_1 = np.stack([idx_a, idx_ab, idx_ac, idx_ad], axis=1)
    tet_2 = np.stack([idx_b, idx_bc, idx_ab, idx_bd], axis=1)
    tet_3 = np.stack([idx_c, idx_ac, idx_bc, idx_cd], axis=1)
    tet_4 = np.stack([idx_d, idx_ad, idx_cd, idx_bd], axis=1)
    tet_5 = np.stack([idx_ab, idx_ac, idx_ad, idx_bd], axis=1)
    tet_6 = np.stack([idx_ab, idx_ac, idx_bd, idx_bc], axis=1)
    tet_7 = np.stack([idx_cd, idx_ac, idx_bd, idx_ad], axis=1)
    tet_8 = np.stack([idx_cd, idx_ac, idx_bc, idx_bd], axis=1)

    tet_list_new_tx8x4 = np.stack(
        [tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], axis=1)

    if tet_list_subdiv_sig is None:
        tet_list_new_Tx4 = tet_list_new_tx8x4.reshape(-1, 4)
    else:
        tet_origin = tet_list_tx4[~tet_list_subdiv_sig]
        tet_new = tet_list_new_tx8x4[tet_list_subdiv_sig]
        tet_list_new_Tx4 = np.concatenate(
            [tet_origin, tet_new.reshape(-1, 4)], axis=0)

    return tet_points_new_Px3, tet_feat_new_Pxk, tet_list_new_Tx4


##########################################################
if __name__ == '__main__':

    points = np.zeros((5, 3), dtype=np.float32)
    feats = np.zeros_like(points)
    tet_list = np.array([[1, 0, 2, 3], [1, 2, 3, 4]], dtype=np.int64)

    from utils_tetsv import tet_adj_share, get_face_use_occ, save_tet_face
    points, tet_list, _ = read_tetrahedron('../data/cube_40_tet.tet')

    filename = '40.obj'
    tet_adj = tet_adj_share(tet_list, points.shape[0])
    tet_p = points[tet_list.reshape(-1, ), :]
    tet_p_1xtx4x3 = tet_p.reshape(1, -1, 4, 3)
    tmpp = get_face_use_occ(tet_p_1xtx4x3,
                            np.ones_like(tet_list[:, :1]),
                            tet_adj,
                            htres=0)
    save_tet_face(tmpp[0], f_name=filename)

    tet_points_new_Px3, tet_feat_new_Px3, tet_list_new_Tx4 = generate_subdivision(
        tet_list, points, points,
        np.random.rand(tet_list.shape[0]) > 0.0)

    filename = '80.obj'
    tet_adj = tet_adj_share(tet_list_new_Tx4, tet_points_new_Px3.shape[0])
    tet_p = tet_points_new_Px3[tet_list_new_Tx4.reshape(-1, ), :]
    tet_p_1xtx4x3 = tet_p.reshape(1, -1, 4, 3)
    tmpp = get_face_use_occ(tet_p_1xtx4x3,
                            np.ones_like(tet_list_new_Tx4[:, :1]),
                            tet_adj,
                            htres=0)
    save_tet_face(tmpp[0], f_name=filename)
