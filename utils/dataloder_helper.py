'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''
import numpy as np
import os
import torch

def get_collate_fn(keys):
    keys = set(keys)

    def collate_fn(data):
        new_data = {}
        for k in data[0].keys():

            if k in keys:
                new_info = tuple(d[k] for d in data)
                new_info = torch.stack(new_info, 0)
            else:
                new_info = tuple(d[k] for d in data)

            new_data[k] = new_info
        return new_data

    return collate_fn

def read_tetrahedron(res=50, root='..'):
    tetrahedrons = []
    vertices = []
    if res > 1.0:
        res = 1.0 / res
    # assert 1 / res == int(1 / res), 'Only support res that divisible to 1'
    root_path = os.path.join(root, 'quartet/meshes')
    file_name = os.path.join(root_path, 'cube_%f_tet.tet' % (res))

    # generate tetrahedron is not exist files
    if not os.path.exists(file_name):
        command = 'cd %s/quartet; ' % (root) + \
                  './quartet meshes/cube.obj %f meshes/cube_%f_tet.tet -s meshes/cube_boundary_%f.obj' % (res, res, res)
        os.system(command)

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
    # import ipdb
    # ipdb.set_trace()
    vertices = np.asarray(vertices)
    vertices[vertices <= (0 + res / 4.0)] = 0  # determine the boundary point
    vertices[vertices >= (1 - res / 4.0)] = 1  # determine the boundary point
    mask = np.logical_and(vertices < 1, vertices > 0)
    return vertices, np.asarray(tetrahedrons), mask
