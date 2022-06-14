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


##############################################
def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def lookatnp(eye_3x1, center_3x1, up_3x1):
    # 3 variables should be length 1
    camz = center_3x1 - eye_3x1
    camz /= np.sqrt(np.sum(camz**2))
    camx = np.cross(camz[:, 0], up_3x1[:, 0]).reshape(3, 1)
    camy = np.cross(camx[:, 0], camz[:, 0]).reshape(3, 1)

    # they are not guaranteed to be 1!!!
    mtx = np.concatenate([unit(camx), unit(camy), -camz], axis=1).transpose()
    shift = -np.matmul(mtx, eye_3x1)
    return mtx, shift


def lookat_batch(param_bx3x3):

    bnum = param_bx3x3.shape[0]
    cam_mat_bx3x3 = []
    cam_pos_bx3 = []

    for i in range(bnum):

        # campos, where camera it is
        # camcenter, where camra is looking at
        # camup, y axis of camera view
        eye_3x1 = param_bx3x3[i][0:1].T
        center_3x1 = param_bx3x3[i][1:2].T
        up_3x1 = param_bx3x3[i][2:3].T

        camviewmtx, camviewshift = lookatnp(eye_3x1, center_3x1, up_3x1)

        # pixel2mesh!!!
        camviewshift = -np.dot(camviewmtx.transpose(), camviewshift)

        cam_mat_bx3x3.append(camviewmtx)
        cam_pos_bx3.append(camviewshift)

    cam_mat_bx3x3 = np.stack(cam_mat_bx3x3, axis=0)
    cam_pos_bx3 = np.stack(cam_pos_bx3, axis=0)

    return cam_mat_bx3x3, cam_pos_bx3


#####################################################
def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3] * np.sin(phi)
    temp = param[3] * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0], dtype=np.float32)
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    # cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])

    # for verify
    # mtx, shift = lookatnp(cam_pos_3xb.reshape(3, 1), np.zeros(shape=(3, 1), dtype=np.float32), np.array([0,1,0], dtype=np.float32).reshape(3, 1))
    # note, it is different from lookatnp
    # new_p = mtx * old_p + shift
    # new_p = cam_mat * (old_p - cam_pos)

    return cam_mat, cam_pos


def camera_info_batch(param_bx4):

    bnum = param_bx4.shape[0]
    cam_mat_bx3x3 = []
    cam_pos_bx3 = []

    for i in range(bnum):
        param = param_bx4[i]
        cam_mat, cam_pos = camera_info(param)
        cam_mat_bx3x3.append(cam_mat)
        cam_pos_bx3.append(cam_pos)

    cam_mat_bx3x3 = np.stack(cam_mat_bx3x3, axis=0)
    cam_pos_bx3 = np.stack(cam_pos_bx3, axis=0)

    return cam_mat_bx3x3, cam_pos_bx3


#####################################################
def perspectiveprojectionnp(fovy, ratio=1.0, near=0.01, far=10.0):

    tanfov = np.tan(fovy / 2.0)
    # top = near * tanfov
    # right = ratio * top
    # mtx = [near / right, 0, 0, 0, \
    #          0, near / top, 0, 0, \
    #          0, 0, -(far+near)/(far-near), -2*far*near/(far-near), \
    #          0, 0, -1, 0]
    mtx = [[1.0 / (ratio * tanfov), 0, 0, 0], \
                [0, 1.0 / tanfov, 0, 0], \
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], \
                [0, 0, -1.0, 0]]
    # return np.array(mtx, dtype=np.float32)
    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]],
                    dtype=np.float32)
