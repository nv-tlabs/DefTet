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
import torch.nn as nn
import torch.optim as optim

import cv2
import numpy as np

############################################
import os
import sys
import time
sys.path.append('..')
from config import rootdir

sys.path.append('%s/2_data' % rootdir)
from load_blender import load_blender_data, preprocess_nerf_blender

sys.path.append('%s/3_model' % rootdir)
from deftet import Deftet


sys.path.append('../5_rendereq')
from deftetrneder import rendermeshcolor, preprocess_save


#####################################################
def create_everying(imfolder, imnamefilter, tetfolder, tetres, tercoef, tetdim,
                    tetdim_fixed):

    # 1) load data
    '''
    imgs_bxhxwx4, cameras_bx3x3, fov = load_blender_data(
        imfolder, ishalf=1, namefilter=imnamefilter)
    fov = np.deg2rad(fov)
    '''
    images, poses, render_poses, hwf, i_split = \
    load_blender_data(imfolder, half_res=True, testskip=8)
    cameras = [poses, render_poses, hwf, i_split]

    # 2) create deftetmodel
    deftetmodel = Deftet(basefolder=tetfolder,
                         res=tetres,
                         coef=tercoef,
                         feature_dim=tetdim,
                         feature_raw=True,
                         feature_fixed_dim=tetdim_fixed,
                         feature_fixed_init=None)

    return images, cameras, deftetmodel


def generatere(deftetmodel,
               tfgtcamrot,
               tfgtcamtrans,
               tfgtcamproj,
               renderfunc,
               viewpoint=False):

    height, width = deftetmodel.get_hw()
    dev = deftetmodel.device

    immask = np.zeros((height, width, 1), dtype=np.float32)
    imcolor = np.zeros((height, width, 3), dtype=np.float32)

    with torch.no_grad():
        step = height
        for i in range(height // step + 1):
            hbe = i * step
            hen = min(height, hbe + step)
            if hbe >= hen:
                break

            impixsample = np.zeros((height, width), dtype=np.bool)
            impixsample[hbe:hen] = True
            tfimpixsample = torch.from_numpy(impixsample).to(dev)

            ###############################################
            imcolor_bxpx3, immask_bxpx1 = deftetmodel(tfimpixsample,
                                                      tfgtcamrot[:1],
                                                      tfgtcamtrans[:1],
                                                      tfgtcamproj,
                                                      renderfunc,
                                                      viewpoint=viewpoint)
            immask[impixsample, :] = immask_bxpx1.detach().cpu().numpy()[0]
            imcolor[impixsample, :] = imcolor_bxpx3.detach().cpu().numpy()[0]

    return immask, imcolor


def optimzie(images,
             cameras,
             deftetmodel,
             lr,
             lr2,
             bs=1,
             optnum=500,
             svfolder='/',
             dev='cpu',
             gridmov=False,
             loadpath=None,
             sublevel=0,
             lossweights=None):
    torch.cuda.synchronize()
    optimize_start_time = time.time()
    prep_start_time = time.time()

    _, height, width, _ = images.shape

    imgs_bxhxwx4 = images
    gtcolor_bxhxwx3 = imgs_bxhxwx4[:, :, :, :3]
    gtmask_bxhxwx1 = imgs_bxhxwx4[:, :, :, 3:4]

    white_bkgd = True
    if white_bkgd:
        gtcolor_bxhxwx3 = gtcolor_bxhxwx3 * gtmask_bxhxwx1 + (1. -
                                                              gtmask_bxhxwx1)

    #################################################################
    poses, render_poses, hwf, i_split = cameras
    camrot_bx3x3, camtrans_bx3x1, camproj_3x1 = preprocess_nerf_blender(
        poses, hwf)

    tfgtcamproj = torch.from_numpy(camproj_3x1).to(dev)

    i_train, i_val, i_test = i_split

    ####################################################
    # 3) optimizer, grid is not moving
    g_params = list(deftetmodel.parameters())[1:]
    g_optim = optim.Adam(g_params, lr, betas=(0.5, 0.999))

    d_params = list(deftetmodel.parameters())[:1]
    d_optim = optim.Adam(d_params, lr2, betas=(0.5, 0.999))

    ################################################################
    prefix = 'fix'
    if gridmov:
        prefix = 'mov'
    imsvfolder = '%s/%s' % (svfolder, prefix)
    if not os.path.isdir(imsvfolder):
        os.mkdir(imsvfolder)
    modelsvfolder = '%s' % (svfolder, )

    if loadpath is not None:
        deftetmodel.load_state_dict(torch.load(loadpath))

    deftetmodel = deftetmodel.to(dev)
    deftetmodel.sethw(height, width, 1000)
    deftetmodel.todev(dev)
    torch.cuda.synchronize()
    print("prep_time: ", time.time() - prep_start_time)
    data_prep_time = 0.
    deftet_model_time = 0.
    loss_time = 0.
    backward_time = 0.
    step_time = 0.
    display_time = 0.
    rest_time = 0.
    ##########################################################
    #with torch.autograd.profiler.profile() as prof:
    if True:
        for i in range(optnum):
            g_optim.zero_grad()
            if gridmov:
                d_optim.zero_grad()
            torch.cuda.synchronize()
            data_prep_start_time = time.time()
            deletenum = args.deletenum
            delthres = args.deletethres
            if (i % deletenum == deletenum - 1) and (i > 0):
                deftetmodel.deletetet(delthres, preprocess_save)
                deftetmodel.to(dev)
                deftetmodel.todev(dev)

            ###################################################
            #
            # print(tfp_px3.grad)
            viewid = np.random.randint(len(i_train), size=(bs, ))
            viewid = i_train[viewid]

            # camera
            camerarot_batch = camrot_bx3x3[viewid]
            camtrans_batch = camtrans_bx3x1[viewid]

            # gt
            gtmask_batch = gtmask_bxhxwx1[viewid]
            gtcolor_batch = gtcolor_bxhxwx3[viewid]

            ####################################################
            pixsam = args.pixelsampling
            impixsample = np.random.rand(height, width).astype(np.float32)
            impixsample = impixsample < pixsam

            gtmask_sample = gtmask_batch[:, impixsample, :]
            gtcolor_sample = gtcolor_batch[:, impixsample, :]

            ##########################################################
            tfgtcamrot = torch.from_numpy(camerarot_batch).to(dev)
            tfgtcamtrans = torch.from_numpy(camtrans_batch).to(dev)

            tfgtmask = torch.from_numpy(gtmask_sample).to(dev)
            tfgtim = torch.from_numpy(gtcolor_sample).to(dev)

            tfimpixsample = torch.from_numpy(impixsample).to(dev)

            data_prep_time += time.time() - data_prep_start_time

            ###############################################
            torch.cuda.synchronize()
            deftet_model_start_time = time.time()
            imcolor_bxpx3, immask_bxpx1 = deftetmodel(tfimpixsample, tfgtcamrot,
                                                      tfgtcamtrans, tfgtcamproj,
                                                      rendermeshcolor)
            torch.cuda.synchronize()
            deftet_model_time += time.time() - deftet_model_start_time
            loss_start_time = time.time()
            ######################################################
            w_im = lossweights['weights_im_loss']
            w_mask = lossweights['weights_mask_loss']
            w_occ = lossweights['weights_mask_reg']
            #w_occ_lap = lossweights['weights_occ_lap']
            #w_color = lossweights['weights_color_reg']

            w_pmov = lossweights['weights_point_mov']
            #w_plap = lossweights['weights_pointlap']
            w_tetvar = lossweights['weights_tetvariance']


            ############################################################3
            loss0 = torch.nn.functional.l1_loss(imcolor_bxpx3, tfgtim)

            loss1 = torch.nn.functional.l1_loss(immask_bxpx1, tfgtmask)

            tffeat_pxd = deftetmodel.get_feat()
            tfweights, tfcolor = preprocess_save(None, tffeat_pxd)
            loss_occ = tfweights.mean()
            loss = loss0 * w_im + loss1 * w_mask + loss_occ * w_occ
            get_featlap_inputs = [tfcolor, tfweights]

            if gridmov:
                points_mov = deftetmodel.get_mov()
                loss_mov = points_mov.abs().mean()
                get_featlap_inputs.append(points_mov)
                #points_lap = deftetmodel.get_featlap(points_mov)
                #loss_lap = (points_lap**2).sum()

                point_variance = deftetmodel.get_volume_variance()
                loss_var = (point_variance**2).sum()

                loss += w_pmov * loss_mov + w_tetvar * loss_var

                #print('weights mov {} weights lap {} weights {}'.format(
                #    w_pmov, w_plap, w_tetvar))
                #print('loss mov {} loss lap {} loss_var {}'.format(
                #    loss_mov, loss_lap, loss_var))
                weights_vector = lossweights['weights_vector_with_gridmov']
            else:
                weights_vector = lossweights['weights_vector']
            get_featlap_output = deftetmodel.get_featlap(torch.cat(get_featlap_inputs, dim=-1))
            loss_vector = torch.sum(get_featlap_output, dim=0)
            loss += torch.dot(loss_vector, weights_vector)


            torch.cuda.synchronize()
            loss_time += time.time() - loss_start_time
            backward_start_time = time.time()
            loss.backward()
            torch.cuda.synchronize()
            backward_time += time.time() - backward_start_time
            step_start_time = time.time()
            g_optim.step()
            if gridmov:
                d_optim.step()
            torch.cuda.synchronize()
            step_time += time.time() - step_start_time
            displaynum = args.i_img

            display_start_time = time.time()
            if i % displaynum == 0:
                imgtnp = gtmask_batch[0]
                imgt2np = gtcolor_batch[0]
                imprenp, impre2np = generatere(deftetmodel, tfgtcamrot,
                                               tfgtcamtrans, tfgtcamproj,
                                               rendermeshcolor)

                if localtrain:
                    cv2.imshow('gt', imgtnp)
                    cv2.imshow('pre', imprenp)
                    cv2.imshow('gt2', imgt2np)
                    cv2.imshow('pre2', impre2np)
                    cv2.waitKey(1)

                im = np.concatenate([imgtnp, imprenp])
                im2 = np.concatenate([imgt2np, impre2np])
                im = np.concatenate([np.tile(im, [1, 1, 3]), im2], axis=1)
                cv2.imwrite("%s/train-%d.png" % (imsvfolder, i), im * 255)
            else:
                if localtrain:
                    imgtnp = gtmask_batch[0]
                    imprenp = np.zeros_like(imgtnp)
                    imprenp[
                        impixsample, :] = immask_bxpx1.detach().cpu().numpy()[0]
                    imgt2np = gtcolor_batch[0]
                    impre2np = np.zeros_like(imgt2np)
                    impre2np[
                        impixsample, :] = imcolor_bxpx3.detach().cpu().numpy()[0]

                    cv2.imshow('gt', imgtnp)
                    cv2.imshow('presample', imprenp)
                    cv2.imshow('gt2', imgt2np)
                    cv2.imshow('pre2sample', impre2np)
                    cv2.waitKey(1)
            torch.cuda.synchronize()
            display_time += time.time() - display_start_time
    rest_start_time = time.time()
    # save model
    torch.save(deftetmodel.state_dict(), '%s/deftet.pth' % modelsvfolder)

    # for test
    mses = []
    psnrs = []
    for i in i_test:
        viewid = [i]
        gtmask_batch = gtmask_bxhxwx1[viewid]
        gtcolor_batch = gtcolor_bxhxwx3[viewid]

        ##########################################################
        camerarot_batch = camrot_bx3x3[viewid]
        camtrans_batch = camtrans_bx3x1[viewid]
        tfgtcamrot = torch.from_numpy(camerarot_batch).to(dev)
        tfgtcamtrans = torch.from_numpy(camtrans_batch).to(dev)
        immasknp, imcolornp = generatere(deftetmodel, tfgtcamrot, tfgtcamtrans,
                                         tfgtcamproj, rendermeshcolor)
        imgtnp = gtmask_batch[0]
        imgt2np = gtcolor_batch[0]
        im = np.concatenate([imgtnp, immasknp])
        im2 = np.concatenate([imgt2np, imcolornp])
        im = np.concatenate([np.tile(im, [1, 1, 3]), im2], axis=1)

        img2mse = lambda x, y: np.mean((x - y)**2)
        mse2psnr = lambda x: -10. * np.log(x) / np.log(10.0)
        mse = img2mse(imgt2np, imcolornp)
        psnr = mse2psnr(mse)
        mses.append(mse)
        psnrs.append(psnr)
        cv2.imwrite(
            "%s/test-%d-mse%.3f-psnr%.3f.png" % (imsvfolder, i, mse, psnr),
            im * 255)
        #print('test mse {} psnr {}'.format(mse, psnr))

    # generate video
    camrot_bx3x3, camtrans_bx3x1, camproj_3x1 = preprocess_nerf_blender(
        render_poses, hwf)
    ims = []
    for i in range(camrot_bx3x3.shape[0]):
        viewid = [i]
        # camera
        camerarot_batch = camrot_bx3x3[viewid]
        camtrans_batch = camtrans_bx3x1[viewid]
        tfgtcamrot = torch.from_numpy(camerarot_batch).to(dev)
        tfgtcamtrans = torch.from_numpy(camtrans_batch).to(dev)
        immasknp, imcolornp = generatere(deftetmodel, tfgtcamrot, tfgtcamtrans,
                                         tfgtcamproj, rendermeshcolor)
        ims.append(imcolornp)

    import imageio
    rgbs = np.stack([(im[:, :, ::-1] * 255).astype(np.uint8) for im in ims],
                    axis=0)
    imageio.mimwrite('%s/rgb-mse%.3f-psnr%.3f.mp4' %
                     (modelsvfolder, np.mean(mses), np.mean(psnrs)),
                     rgbs,
                     fps=30,
                     quality=8)

    # save obj
    deftetmodel.saveobj(savedir=modelsvfolder,
                        prefix=prefix,
                        processfunc=preprocess_save)
    torch.cuda.synchronize()
    rest_time += time.time() - rest_start_time

    print("optimize_time: ", time.time() - optimize_start_time)
    print("data_prep_time: ", data_prep_time)
    print("deftet_model_time: ", deftet_model_time)
    print("loss_time: ", loss_time)
    print("backward_time: ", backward_time)
    print("step_time: ", step_time)
    print("display_time: ", display_time)
    print("rest_time: ", rest_time)

######################################################
if __name__ == '__main__':
    from expconfig import config_parser
    parser = config_parser()
    args = parser.parse_args()

    localtrain = not args.remote

    expname = args.expname
    datadir = '%s/%s' % (args.datadir, expname)
    imnamefilter = None  # 'r16'

    tetfolder = '%s/data' % rootdir
    tetres = args.tetres
    tetcoef = args.tetcoef
    tetdim = args.tetdim
    tetdim_fixed = args.tetdim_fixed
    create_start_time = time.time()
    images, cameras, deftetmodel = create_everying(datadir, imnamefilter,
                                                   tetfolder, tetres, tetcoef,
                                                   tetdim, tetdim_fixed)
    print("create_time: ", time.time() - create_start_time)

    ####################################################
    # training parameters
    from collections import OrderedDict
    weights = OrderedDict()

    weights['sublevel'] = args.sublevel
    weights['optnum_fix'] = args.optfixnum
    weights['optnum_mov'] = args.optmovnum

    weights['lr_fix'] = args.lrfix
    weights['lr_mov'] = args.lrmov

    weights['weights_im_loss'] = args.weights_im_loss
    weights['weights_mask_loss'] = args.weights_mask_loss
    weights['weights_mask_reg'] = args.weights_mask_reg
    #weights['weights_occ_lap'] = args.weights_occ_lap
    #weights['weights_color_reg'] = args.weights_color_reg

    weights['weights_point_mov'] = args.weights_point_mov
    #weights['weights_pointlap'] = args.weights_pointlap
    weights['weights_tetvariance'] = args.weights_tetvariance

    weights['weights_vector'] = torch.tensor([
        args.weights_color_reg,
        args.weights_color_reg,
        args.weights_color_reg,
        args.weights_occ_lap
    ], device='cuda', requires_grad=False, dtype=torch.float)
    weights['weights_vector_with_gridmov'] = torch.tensor([
        args.weights_color_reg,
        args.weights_color_reg,
        args.weights_color_reg,
        args.weights_occ_lap,
        args.weights_point_mov,
        args.weights_point_mov,
        args.weights_point_mov
    ], device='cuda', requires_grad=False, dtype=torch.float)
    ###############################################################
    svfolder = '%s/%s-tet%d-dim%d_%d' % (args.savedir, expname, tetres, tetdim,
                                         tetdim_fixed)
    if os.path.isdir(svfolder):
        sig = 0
        while True:
            sig += 1
            svfolder_tmp = '%s-%d' % (svfolder, sig)
            if not os.path.isdir(svfolder_tmp):
                break
        svfolder = svfolder_tmp
    '''
    cmd = 'rm -fr %s' % svfolder
    os.system(cmd)
    '''
    os.makedirs(svfolder)

    ###############################################
    sublevel = weights['sublevel']
    optnum_fix = weights['optnum_fix']
    optnum_mov = weights['optnum_mov']
    lr_fix = weights['lr_fix']
    lr_mov = weights['lr_mov']
    print("Start_training:")
    for i in range(sublevel + 1):
        for gridmov in [True, False]: #[False, True]:
            print("level: {} ({})".format(i, gridmov))

            if (not gridmov):
                svfolder2 = '%s/stage-sub_%d-fix_%d' % (svfolder, i, 0)
                optnum = optnum_fix
                lr1 = lr_fix / (i + 1)
                lr2 = lr_mov / (i + 1)
            else:
                svfolder2 = '%s/stage-sub_%d-carve_%d' % (svfolder, i, 0)
                optnum = optnum_mov
                lr1 = lr_fix / (i + 1)
                lr2 = lr_mov / (i + 1)

            if not os.path.isdir(svfolder2):
                os.mkdir(svfolder2)

            optimzie(images,
                     cameras,
                     deftetmodel,
                     lr=lr1,
                     lr2=lr2,
                     bs=1,
                     optnum=optnum,
                     svfolder=svfolder2,
                     dev='cuda',
                     gridmov=gridmov,
                     loadpath=None,
                     sublevel=i,
                     lossweights=weights)

        if i < sublevel:
            # load from fixed tet
            deftetmodel.subdivision(None)

