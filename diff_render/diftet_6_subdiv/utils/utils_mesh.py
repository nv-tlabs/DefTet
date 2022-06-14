'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

from __future__ import print_function
from __future__ import division

import os
import cv2
import torch
import numpy as np


##################################################################
# faces begin from 0!!!
def face2edge(facenp_fx3):
    '''
    facenp_fx3, int32
    return edgenp_ex2, int32
    '''
    f1 = facenp_fx3[:, 0:1]
    f2 = facenp_fx3[:, 1:2]
    f3 = facenp_fx3[:, 2:3]
    e1 = np.concatenate((f1, f1, f2), axis=0)
    e2 = np.concatenate((f2, f3, f3), axis=0)
    edgenp_ex2 = np.concatenate((e1, e2), axis=1)
    # sort & unique
    edgenp_ex2 = np.sort(edgenp_ex2, axis=1)
    edgenp_ex2 = np.unique(edgenp_ex2, axis=0)
    return edgenp_ex2


def edge2face(facenp_fx3, edgenp_ex2):
    '''
    facenp_fx3, int32
    edgenp_ex2, int32
    return edgenp_ex2, int32
    this edge is indexed by face
    '''
    fnum = facenp_fx3.shape[0]
    enum = edgenp_ex2.shape[0]

    facesort = np.sort(facenp_fx3, axis=1)
    edgesort = np.sort(edgenp_ex2, axis=1)
    edgere_ex2 = -np.ones_like(edgesort)

    from tqdm import tqdm
    for i in tqdm(range(enum)):
        pbe, pen = edgesort[i]
        eid = 0
        for j in range(fnum):
            f1, f2, f3 = facesort[j]
            cond1 = f1 == pbe and f2 == pen
            cond2 = f1 == pbe and f3 == pen
            cond3 = f2 == pbe and f3 == pen
            if cond1 or cond2 or cond3:
                edgere_ex2[i, eid] = j
                eid += 1

    return edgere_ex2


def edge2face_multi(facenp_fx3, edgenp_ex2):
    '''
    facenp_fx3, int32
    edgenp_ex2, int32
    return edgenp_exk, int32
    this edge is indexed by face
    '''
    fnum = facenp_fx3.shape[0]
    enum = edgenp_ex2.shape[0]

    facesort = np.sort(facenp_fx3, axis=1)
    f1, f2, f3 = facesort[:, 0], facesort[:, 1], facesort[:, 2],

    edgesort = np.sort(edgenp_ex2, axis=1)
    edgere_exk = -np.ones((len(edgesort), 30), dtype=np.int64)
    emax = -1

    from tqdm import tqdm
    for i in tqdm(range(enum)):
        pbe, pen = edgesort[i]

        cond1 = (f1 == pbe) & (f2 == pen)
        cond2 = (f1 == pbe) & (f3 == pen)
        cond3 = (f2 == pbe) & (f3 == pen)
        cond = cond1 | cond2 | cond3
        idx = np.where(cond)[0]
        edgere_exk[i, :len(idx)] = idx

        if emax < len(idx):
            emax = len(idx)

    assert emax < 30
    print('emax', emax)
    edgere_exk = edgere_exk[:, :emax]

    return edgere_exk


#####################################################
def face2pneimtx(facenp_fx3, pnum):
    '''
    facenp_fx3, int32
    return pointneighbourmtx, pxp, float32
    will normalize!
    assume it is a good mesh
    every point has more than one neighbour
    '''
    # pnum = np.max(facenp_fx3) + 1
    pointneighbourmtx = np.zeros(shape=(pnum, pnum), dtype=np.float32)
    for i in range(3):
        be = i
        en = (i + 1) % 3
        idx1 = facenp_fx3[:, be]
        idx2 = facenp_fx3[:, en]
        pointneighbourmtx[idx1, idx2] = 1
        pointneighbourmtx[idx2, idx1] = 1
    pointneicount = np.sum(pointneighbourmtx, axis=1, keepdims=True)
    assert np.all(pointneicount > 0)
    pointneighbourmtx /= pointneicount
    return pointneighbourmtx


def face2pfmtx(facenp_fx3, pnum):
    '''
    facenp_fx3, int32
    reutrn pfmtx, pxf, float32
    '''
    # pnum = np.max(facenp_fx3) + 1
    fnum = facenp_fx3.shape[0]
    pfmtx = np.zeros(shape=(pnum, fnum), dtype=np.float32)
    for i, f in enumerate(facenp_fx3):
        pfmtx[f[0], i] = 1
        pfmtx[f[1], i] = 1
        pfmtx[f[2], i] = 1
    return pfmtx


################################################################
def loadobj(meshfile):

    v = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3


def loadobjtex(meshfile):

    v = []
    vt = []
    f = []
    ft = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:]])
        if data[0] == 'vt':
            vt.append([float(d) for d in data[1:]])
        if data[0] == 'f':
            data = [da.split('/') for da in data]
            f.append([int(d[0]) for d in data[1:]])
            ft.append([int(d[1]) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    ftnp_fx3 = np.array(ft, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    uvs = np.array(vt, dtype=np.float32)[:, :2]
    return pointnp_px3, facenp_fx3, uvs, ftnp_fx3


def savemesh(pointnp_px3, facenp_fx3, fname, inverse=False):

    fid = open(fname, 'w')
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
    for f in facenp_fx3:
        f1 = f + 1
        if not inverse:
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        else:
            fid.write('f %d %d %d\n' % (f1[0], f1[2], f1[1]))
    fid.close()
    return


def savemeshcolor(pointnp_px3, facenp_fx3, fname, color_px3=None):

    fid = open(fname, 'w')
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        tetocc = color_px3[pidx]
        fid.write('v %f %f %f %f %f %f\n' %
                  (pp[0], pp[1], pp[2], tetocc[0], tetocc[1], tetocc[2]))
    for f in facenp_fx3:
        f1 = f + 1
        fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
    fid.close()

    return


##########################################################
def savemeshfweights(pointnp_px3, pointweights_fx1, facenp_fx3, fname, thres):

    fid = open(fname, 'w')
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
    for i, f in enumerate(facenp_fx3):
        pw = pointweights_fx1[i, 0]
        if pw > thres:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
    fid.close()
    return


def savemeshfweightscolor(pointnp_px3,
                          pointweights_fx1,
                          facenp_fx3,
                          fname,
                          thres,
                          color_px3=None):

    fid = open(fname, 'w')
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        tetocc = color_px3[pidx]
        fid.write('v %f %f %f %f %f %f\n' %
                  (pp[0], pp[1], pp[2], tetocc[0], tetocc[1], tetocc[2]))
    for i, f in enumerate(facenp_fx3):
        pw = pointweights_fx1[i, 0]
        if pw > thres:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
    fid.close()
    return


###################################################################33
def savemeshtes(pointnp_px3, tcoords_px2, facenp_fx3, fname):

    import os
    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = '%s/%s.mtl' % (fol, na)
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write('Kd 1 1 1\n')
    fid.write('Ka 0 0 0\n')
    fid.write('Ks 0.4 0.4 0.4\n')
    fid.write('Ns 10\n')
    fid.write('illum 2\n')
    fid.write('map_Kd %s.png\n' % na)
    fid.close()

    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for f in facenp_fx3:
        f1 = f + 1
        fid.write('f %d/%d %d/%d %d/%d\n' %
                  (f1[0], f1[0], f1[1], f1[1], f1[2], f1[2]))
    fid.close()

    return


def saveobjscale(meshfile, scale, maxratio, shift=None):

    mname, prefix = os.path.splitext(meshfile)
    mnamenew = '%s-%.2f%s' % (mname, maxratio, prefix)

    meshfp = open(meshfile, 'r')
    meshfp2 = open(mnamenew, 'w')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            meshfp2.write(line)
            continue
        else:
            if data[0] == 'v':
                p = [scale * float(d) for d in data[1:]]
                meshfp2.write('v %f %f %f\n' % (p[0], p[1], p[2]))
            else:
                meshfp2.write(line)
                continue

    meshfp.close()
    meshfp2.close()

    return


################################################################3
if __name__ == '__main__':

    meshjson = '1.obj'

    # f begin from 0!!!
    pointnp_px3, facenp_fx3 = loadobj(meshjson)
    assert np.max(facenp_fx3) == pointnp_px3.shape[0] - 1
    assert np.min(facenp_fx3) == 0

    pointnp_px3[:, 1] -= 0.05
    X = pointnp_px3[:, 0]
    Y = pointnp_px3[:, 1]
    Z = pointnp_px3[:, 2]
    h = 248 * (Y / Z) + 111.5
    w = -248 * (X / Z) + 111.5

    height = 224
    width = 224
    im = np.zeros(shape=(height, width), dtype=np.uint8)
    for cir in zip(w, h):
        cv2.circle(im, (int(cir[0]), int(cir[1])), 3, (255, 0, 0), -1)
    cv2.imshow('', im)
    cv2.waitKey()

    # edge, neighbour and pfmtx
    edgenp_ex2 = face2edge(facenp_fx3)

    face_edgeidx_fx3 = face2edge2(facenp_fx3, edgenp_ex2)

    pneimtx = face2pneimtx(facenp_fx3)
    pfmtx = face2pfmtx(facenp_fx3)

    # save
    savemesh(pointnp_px3, facenp_fx3, '1s.obj')
