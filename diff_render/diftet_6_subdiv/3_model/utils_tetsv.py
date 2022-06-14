'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
from __future__ import print_function
from __future__ import division

import numpy as np
import tqdm


#############################################################
def tet_adj_share(tet_list_tx4, n_point):
    '''
    # convet tet 2 mesh
    '''
    n_tet = tet_list_tx4.shape[0]
    # using sparse matric
    index_list = []
    idx_array = [0, 1, 2, 1, 0, 3, 2, 3, 0, 3, 2, 1]
    idx_array = np.asarray(idx_array).reshape(4, 3)

    ########################################################
    # A naive implementation version.  You need to check cuda again!!!! some problem will happen for this!!!!
    face_index = dict()
    for tet_idx, tet in enumerate(tet_list_tx4):
        for i_face in range(4):
            idx = tet[idx_array[i_face][0]], tet[idx_array[i_face][1]], tet[
                idx_array[i_face][2]]
            a, c, b = sorted(idx)

            num = a * n_point * n_point + b * n_point + c
            if (num not in face_index):
                face_index[num] = [[tet_idx, i_face]]
            else:
                face_index[num].append([tet_idx, i_face])

    singular_face = 0
    tet_neighbour_idx = -np.ones((n_tet, 4), dtype=np.int64)
    tet_neighbour_idx_len = np.zeros((n_tet, ), dtype=np.int64)
    # check the face index for each tet
    for face_num in face_index.keys():
        all_tet = face_index[face_num]
        if len(all_tet) == 1:
            singular_face += 1
        if len(all_tet) == 2:
            index_list.append([all_tet[0][0], all_tet[1][0], all_tet[0][1]])
            index_list.append([all_tet[1][0], all_tet[0][0], all_tet[1][1]])
            tet0 = all_tet[0][0]
            tet1 = all_tet[1][0]
            assert tet_neighbour_idx_len[tet0] <= 3
            assert tet_neighbour_idx_len[tet1] <= 3
            tet_neighbour_idx[tet0, tet_neighbour_idx_len[tet0]] = tet1
            tet_neighbour_idx_len[tet0] += 1
            tet_neighbour_idx[tet1, tet_neighbour_idx_len[tet1]] = tet0
            tet_neighbour_idx_len[tet1] += 1
        if len(all_tet) > 2:
            raise ValueError
    index_list = np.asarray(index_list)
    index_value = np.ones(index_list.shape[0])

    ###########################################################
    from scipy.sparse import coo_matrix
    adj_list = []
    for i in range(4):
        adj_list.append(
            coo_matrix((index_value[index_list[:, 2] == i],
                        (index_list[:, 0][index_list[:, 2] == i],
                         index_list[:, 1][index_list[:, 2] == i])),
                       shape=(n_tet, n_tet)))

    return adj_list, tet_neighbour_idx


###########################################################################
def get_face_use_occ(tet_bxfx4x3, center_occ_cuda, tet_adj, htres=0.25):
    # tet_bxfx4x3: bxtx4x3
    # center_occ_cuda: tx1
    # tet_adj: get it by function  get_tet_adj()
    batch_size = tet_bxfx4x3.shape[0]
    assert batch_size == 1

    all_equal_list = []

    for i in range(4):

        neibor_occ = tet_adj[i].dot(center_occ_cuda)
        # have different neighbor
        equal = np.abs(neibor_occ - center_occ_cuda) > htres

        equal = equal & (center_occ_cuda > htres * 2)
        '''
        has_adj = torch.sparse.sum(tet_adj[i], dim=1).to_dense().bool().unsqueeze(0).unsqueeze(-1)
        equal = equal & has_adj
        '''
        all_equal_list.append(equal)
    # all_equal_list = torch.cat(all_equal_list, dim=-1)
    all_equal_list = np.concatenate(all_equal_list, axis=-1)

    A = tet_bxfx4x3[:, :, 0:1, :]
    B = tet_bxfx4x3[:, :, 1:2, :]
    C = tet_bxfx4x3[:, :, 2:3, :]
    D = tet_bxfx4x3[:, :, 3:4, :]

    all_face_a = np.concatenate([A, B, C, D], axis=2)
    all_face_b = np.concatenate([B, A, D, C], axis=2)
    all_face_c = np.concatenate([C, D, A, B], axis=2)
    all_equal_list = np.tile(np.expand_dims(all_equal_list, axis=-1),
                             [1, 1, 3])

    # all_equal_list_np = all_equal_list.data.cpu().numpy()
    all_face = []
    for i_batch in range(batch_size):
        equal = all_equal_list
        face_a = all_face_a[i_batch][equal]
        face_a = face_a.reshape(-1, 1, 3)
        face_b = all_face_b[i_batch][equal]
        face_b = face_b.reshape(-1, 1, 3)
        face_c = all_face_c[i_batch][equal]
        face_c = face_c.reshape(-1, 1, 3)
        face = np.concatenate([face_a, face_b, face_c], axis=1)
        all_face.append(face)
        # face: fx3x3

    return all_face


def save_tet_face(tet_fx3x3, f_name):
    # use trimesh to fix the surface
    with open(f_name, 'w') as f:
        all_str = ''
        for idx_tri, triangle in enumerate(tet_fx3x3):
            for i in range(3):
                all_str += 'v %f %f %f\n' % (triangle[i][0], triangle[i][1],
                                             triangle[i][2])
            idx = idx_tri * 3
            all_str += 'f %d %d %d\n' % (idx + 1, idx + 3, idx + 2)
        f.write(all_str)


################################################################################
def get_face_use_occ_color(tet_bxfx4x3,
                           tetcolor_bxfx4x3,
                           center_occ_cuda,
                           tet_adj,
                           htres=0.25):
    # tet_bxfx4x3: bxtx4x3
    # center_occ_cuda: tx1
    # tet_adj: get it by function  get_tet_adj()
    batch_size = tet_bxfx4x3.shape[0]
    assert batch_size == 1

    all_equal_list = []

    for i in range(4):

        neibor_occ = tet_adj[i].dot(center_occ_cuda)
        # have different neighbor
        equal = np.abs(neibor_occ - center_occ_cuda) > htres

        equal = equal & (center_occ_cuda > htres * 2)
        '''
        has_adj = torch.sparse.sum(tet_adj[i], dim=1).to_dense().bool().unsqueeze(0).unsqueeze(-1)
        equal = equal & has_adj
        '''
        all_equal_list.append(equal)
    # all_equal_list = torch.cat(all_equal_list, dim=-1)
    all_equal_list = np.concatenate(all_equal_list, axis=-1)

    A = tet_bxfx4x3[:, :, 0:1, :]
    B = tet_bxfx4x3[:, :, 1:2, :]
    C = tet_bxfx4x3[:, :, 2:3, :]
    D = tet_bxfx4x3[:, :, 3:4, :]

    Acolor = tetcolor_bxfx4x3[:, :, 0:1, :]
    Bcolor = tetcolor_bxfx4x3[:, :, 1:2, :]
    Ccolor = tetcolor_bxfx4x3[:, :, 2:3, :]
    Dcolor = tetcolor_bxfx4x3[:, :, 3:4, :]

    all_face_a = np.concatenate([A, B, C, D], axis=2)
    all_face_b = np.concatenate([B, A, D, C], axis=2)
    all_face_c = np.concatenate([C, D, A, B], axis=2)

    all_face_a_color = np.concatenate([Acolor, Bcolor, Ccolor, Dcolor], axis=2)
    all_face_b_color = np.concatenate([Bcolor, Acolor, Dcolor, Ccolor], axis=2)
    all_face_c_color = np.concatenate([Ccolor, Dcolor, Acolor, Bcolor], axis=2)

    all_equal_list = np.tile(np.expand_dims(all_equal_list, axis=-1),
                             [1, 1, 3])

    # all_equal_list_np = all_equal_list.data.cpu().numpy()
    all_face = []
    all_face_color = []
    for i_batch in range(batch_size):
        equal = all_equal_list
        face_a = all_face_a[i_batch][equal]
        face_a = face_a.reshape(-1, 1, 3)

        face_a_color = all_face_a_color[i_batch][equal]
        face_a_color = face_a_color.reshape(-1, 1, 3)

        face_b = all_face_b[i_batch][equal]
        face_b = face_b.reshape(-1, 1, 3)

        face_b_color = all_face_b_color[i_batch][equal]
        face_b_color = face_b_color.reshape(-1, 1, 3)

        face_c = all_face_c[i_batch][equal]
        face_c = face_c.reshape(-1, 1, 3)

        face_c_color = all_face_c_color[i_batch][equal]
        face_c_color = face_c_color.reshape(-1, 1, 3)

        face = np.concatenate([face_a, face_b, face_c], axis=1)
        face_color = np.concatenate([face_a_color, face_b_color, face_c_color],
                                    axis=1)

        all_face.append(face)
        all_face_color.append(face_color)
        # face: fx3x3

    return all_face, all_face_color


def save_tet_face_color(tet_fx3x3, tetcolor_fx3x3, f_name):
    # use trimesh to fix the surface
    with open(f_name, 'w') as f:
        all_str = ''
        for idx_tri, triangle in enumerate(tet_fx3x3):
            trianglecolor = tetcolor_fx3x3[idx_tri]
            for i in range(3):
                all_str += 'v %f %f %f %f %f %f\n' % (triangle[i][0], triangle[i][1], triangle[i][2], \
                                                      trianglecolor[i][0], trianglecolor[i][1], trianglecolor[i][2])
            idx = idx_tri * 3
            all_str += 'f %d %d %d\n' % (idx + 1, idx + 3, idx + 2)
        f.write(all_str)
