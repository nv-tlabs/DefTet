'''
MIT License

Copyright (c) 2020 bmild

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t: torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t],
                                  [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor(
    [[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0],
     [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor(
    [[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0],
     [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]
                  ])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)),
                  'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(cv2.imread(fname, -1))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(
            np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([
        pose_spherical(angle, -30.0, 4.0)
        for angle in np.linspace(-180, 180, 40 + 1)[:-1]
    ], 0)
    render_poses = render_poses.detach().cpu().numpy()

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W),
                                          interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


'''

def p2f_nerfdata(points_px3, c2w, HWF):
    
    # perspective
    # P_W = R * P_C + T
    # thus
    # P_C = R' * (P_W - T)
    R_c2w_3x3 = c2w[:3, :3]
    T_c2w_3x1 = c2w[:3, 3:]
    
    points_world_3xp = points_px3.permute(1, 0)
    Rt_w2c_3x3 = R_c2w_3x3.permute(1, 0)
    points_cam_3xp = torch.matmul(Rt_w2c_3x3, points_world_3xp - T_c2w_3x1)
    
    # next, proj it on image
    xy_2xp = points_cam_3xp[:2]
    z_1xp = -points_cam_3xp[2:3]
    xyproj_2xp = xy_2xp / z_1xp
    
    # next, put it on screen
    h, w, f = HWF
    # assert h == w
    
    #  (x, y, z) = (iii * H *.5 / focal, jjj * H *.5 / focal, -1)
    # (iii, jjj) = (x * focal / (H * .5), y * focal / (H * .5))
    xyscreen_2xp = xyproj_2xp * f / (h / 2)
    
    #  (iii, jjj) = (ii  * W/H, jj / * H/H)
    # (ii, jj) = (iii * h / w, jjj)
    xyscreen_2xp[:1] = xyscreen_2xp[:1] * h / w;
    if DEBUG:
        print(xyscreen_2xp.min(1))
        print(xyscreen_2xp.max(1))
    
    points_cam_1xpx3 = points_cam_3xp.permute(1, 0).unsqueeze(0)
    xyscreen_1xpx2 = xyscreen_2xp.permute(1, 0).unsqueeze(0)
    
    return points_cam_1xpx3, xyscreen_1xpx2
    '''
'''
def perspective(points_bxpx3, cameras):
    # perspective, use just one camera
    camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras
    cameratrans_rot_bx3x3 = camera_rot_bx3x3.permute(0, 2, 1)

    # follow pixel2mesh!!!
    # new_p = cam_mat * (old_p - cam_pos)
    points_bxpx3 = points_bxpx3 - camera_pos_bx3.view(-1, 1, 3)
    points_bxpx3 = torch.matmul(points_bxpx3, cameratrans_rot_bx3x3)

    camera_proj_bx1x3 = camera_proj_3x1.view(-1, 1, 3)
    xy_bxpx3 = points_bxpx3 * camera_proj_bx1x3
    xy_bxpx2 = xy_bxpx3[:, :, :2] / xy_bxpx3[:, :, 2:3]

    return points_bxpx3, xy_bxpx2
    '''


def preprocess_nerf_blender(render_poses_bx4x4, hwf_3):

    camrot_c2w_bx3x3 = render_poses_bx4x4[:, :3, :3]
    camtrans_c2w_bx3x1 = render_poses_bx4x4[:, :3, 3:]
    assert (np.all(render_poses_bx4x4[:, 3, :3] == 0))
    assert (np.all(render_poses_bx4x4[:, 3, 3] == 1))

    # P_w = R_c2w * P_c + T
    # P_c = R_c2w' * (P_w - T)
    camrot_w2c_bx3x3 = np.transpose(camrot_c2w_bx3x3, [0, 2, 1])
    # T will not change
    camtrans_c2w_bx3x1 = camtrans_c2w_bx3x1

    H, W, focal = hwf_3
    camproj_3x1 = np.array([focal / W * 2, focal / H * 2, -1],
                           dtype=np.float32).reshape(-1, 1)

    # assume we have a 3D point (X, Y, Z) in camera coordinate
    # we first project it on image plane
    # x, y = X/-Z, Y/-Z
    # we then draw it on screen
    # consider that the screen with H, W, F
    # we set the image plane as f is 1, thus
    # h_half = H/2/F, w_half = W/2/F
    # finally, we need to convert h_half and w_half to screen coordinate
    # u, v = x/w_half, y/h_half
    # lastly, we convert u, v to pixel coordinate (i, j)
    # i, j = H-(v+1)/2*H,  (u+1)/2*W

    # Now we want to inverse it
    # for pixel (i, j), i in [0, H), j in [0, W)
    # we first normalize it
    # -v, u = (i + 0.5)/H*2-1, (j+0.5)/W*2-1
    # we then map them in image plane
    # x, y = u*(W/2/F), v*(H/2/F)

    # thus, the projection matrix is
    # [F/(W/2), F/(H/2), -1]
    return camrot_w2c_bx3x3, camtrans_c2w_bx3x1, camproj_3x1