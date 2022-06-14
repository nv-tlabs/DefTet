'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
import kaolin as kal

esp = 1e-15
def iou(points1: torch.Tensor, points2: torch.Tensor, thresh=.5):
    r""" Computes the intersection over union values for two sets of points

    Args:
            points1 (torch.Tensor): first points
            points2 (torch.Tensor): second points
    Returns:
            iou (torch.Tensor) : IoU scores for the two sets of points

    Examples:
            >>> points1 = torch.rand( 1000)
            >>> points2 = torch.rand( 1000)
            >>> loss = iou(points1, points2)
            tensor(0.3400)
    """
    points1 = points1.clone()
    points2 = points2.clone()
    points1[points1 <= thresh] = 0
    points1[points1 > thresh] = 1

    points2[points2 <= thresh] = 0
    points2[points2 > thresh] = 1

    points1 = points1.view(-1).byte()
    points2 = points2.view(-1).byte()

    assert points1.shape == points2.shape, 'points1 and points2 must have the same shape'

    iou = torch.sum(torch.mul(points1, points2).float()) / \
        torch.sum((points1 + points2).clamp(min=0, max=1).float())

    return iou



def hausdorff_distance(mesh_a_v, mesh_a_f, mesh_b_v, mesh_b_f, pts_a, pts_b):
    face_vertices = kal.ops.mesh.index_vertices_by_faces(mesh_a_v.unsqueeze(dim=0), mesh_a_f)
    squared_distances_a, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(pts_b.unsqueeze(dim=0), face_vertices)

    face_vertices = kal.ops.mesh.index_vertices_by_faces(mesh_b_v.unsqueeze(dim=0), mesh_b_f)
    squared_distances_b, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(pts_a.unsqueeze(dim=0), face_vertices)

    avg_distance = torch.mean((torch.sqrt(squared_distances_a + esp) + torch.sqrt(squared_distances_b + esp)) / 2)

    max_distance = (torch.max(torch.sqrt(squared_distances_a + esp)) + torch.max(torch.sqrt(squared_distances_b + esp))) /2
    if torch.isnan(avg_distance).any():
        import ipdb
        ipdb.set_trace()
        print('find nan')
    return avg_distance, max_distance




def f_score(gt_points: torch.Tensor, pred_points: torch.Tensor,
            radius: float = 0.01, extend=False):
    r""" Computes the f-score of two sets of points, with a hit defined by two point existing withing a defined radius of each other

    Args:
            gt_points (torch.Tensor): ground truth points
            pred_points (torch.Tensor): predicted points points
            radius (float): radisu from a point to define a hit
            extend (bool): if the alternate f-score definition should be applied

    Returns:
            (float): computed f-score

    Example:
            >>> points1 = torch.rand(1000)
            >>> points2 = torch.rand(1000)
            >>> loss = f_score(points1, points2)
            >>> loss
            tensor(0.0070)

    """

    pred_distances = torch.sqrt(kal.metrics.pointcloud.sided_distance(gt_points, pred_points)[0] + esp)
    gt_distances = torch.sqrt(kal.metrics.pointcloud.sided_distance(pred_points, gt_points)[0]+ esp)

    if extend:
        fp = (gt_distances > radius).float().sum()
        tp = (gt_distances <= radius).float().sum()
        precision = tp / (tp + fp)
        tp = (pred_distances <= radius).float().sum()
        fn = (pred_distances > radius).float().sum()
        recall = tp / (tp + fn)

    else:
        fn = torch.sum(pred_distances > radius)
        fp = torch.sum(gt_distances > radius).float()
        tp = torch.sum(gt_distances <= radius).float()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f_score

def chamfer_distance(S1, S2):
    dist_to_S2 = torch.sqrt(kal.metrics.pointcloud.sided_distance(S1, S2)[0] + esp)
    dist_to_S1 = torch.sqrt(kal.metrics.pointcloud.sided_distance(S2, S1)[0] + esp)

    distance = (dist_to_S2.mean() +  dist_to_S1.mean()) / 2
    return distance


def chamfer_distance_l1(S1, S2):
    _, idx1 = kal.metrics.pointcloud.sided_distance(S1, S2)

    closest_S2 = torch.index_select(S2[0], 0, idx1[0])#

    dist_to_S2 = ((torch.abs(S1 - closest_S2.unsqueeze(dim=0))).sum(dim=-1))

    _, idx2 = kal.metrics.pointcloud.sided_distance(S2, S1)
    closest_S1 = torch.index_select(S1[0], 0, idx2[0])
    dist_to_S1 = ((torch.abs(S2 - closest_S1.unsqueeze(dim=0))).sum(dim=-1))

    return dist_to_S2.mean() + dist_to_S1.mean()






