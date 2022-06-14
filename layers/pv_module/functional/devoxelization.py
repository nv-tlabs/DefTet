from torch.autograd import Function
import torch
from layers.pv_module.functional.backend import _backend
import torch.nn.functional as F

__all__ = ['trilinear_devoxelize', 'trilinear_devoxelize_ori']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1).float()
        coords = coords.contiguous().float()
        outs, inds, wgts = _backend.trilinear_devoxelize_forward(resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = _backend.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None

trilinear_devoxelize_ori =  TrilinearDevoxelization.apply

def trilinear_devoxelize(c, coords, r, training=None):
    coords = (coords * 2 + 1.0) / r - 1.0
    coords = coords.permute(0, 2, 1).reshape(c.shape[0], 1, 1, -1, 3)
    coords = torch.flip(coords, dims=[-1])
    f = F.grid_sample(input=c, grid=coords, padding_mode='border', align_corners=False)
    f = f.squeeze(dim=2).squeeze(dim=2)
    return f