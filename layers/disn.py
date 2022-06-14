'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.vgg import VGG, cfgs
from copy import deepcopy

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    '''
    # import ipdb
    # ipdb.set_trace()
    x = x.clone()
    # assert x.shape[1] == 3
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

class VGG16WithFeatures(VGG):

    @staticmethod
    def make_layers(config, in_channels=3):
        layers = []
        # in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.ModuleList(layers)

    def __init__(self, pretrained, num_classes, in_channels=3):
        super().__init__(VGG16WithFeatures.make_layers(cfgs['D'], in_channels=in_channels),
                         num_classes=num_classes,
                         )

        if pretrained:
            self.my_load_state_dict(model_zoo.load_url(
                'https://download.pytorch.org/models/vgg16-397923af.pth'),
                strict=True)

    def my_load_state_dict(self, state_dict, strict=True):
        # copy first conv
        new_state_dict = {}
        for k in state_dict:
            if k == 'features.0.weight':
                #########################
                new_v = torch.zeros_like(self.features[0].weight.data)
                old_v = state_dict[k]
                new_v[:, :old_v.shape[1], :, :] = deepcopy(old_v)
                for i in range(old_v.shape[1], new_v.shape[1]):
                    new_v[:, i, :, :] = deepcopy(old_v[:, -1, :, :])
            else:
                new_v = state_dict[k]
            new_state_dict[k] = new_v

        self.load_state_dict(new_state_dict, strict=strict)

    def forward(self, x):
        intermediate = []
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                intermediate.append(x)

            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, intermediate


def make_mlp(dims, activation=nn.ReLU, batch_norm=False):
    """
    Create an MLP for SDF decoder etc.
    NOTE: the input needs to have shape (batch_size, channels, dim).

    Args:
        dims: A list of dimensions.
        activation: The activation function to use.
        batch_norm: Whether to apply batch normalization.
    """
    assert len(dims) >= 2
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Conv1d(dims[i], dims[i + 1], kernel_size=1, bias=True))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        if activation:
            layers.append(activation())

    return nn.Sequential(*layers)


class SDFGlobalDecoder(nn.Module):

    def __init__(self, out_features, batch_norm=False):
        super().__init__()

        self.mlp1 = make_mlp([3, 64, 256, 512], batch_norm=batch_norm)
        # TODO: Get the actual dimension of the feats
        self.mlp2 = make_mlp([1512, 512, 256], batch_norm=batch_norm)
        self.mlp3 = make_mlp([256, out_features], activation=None,
                             batch_norm=False)

    def forward(self, query_points, global_features):
        """
        Args:
            query_points: (batch_size, num_points, 3)
            global_features: (batch_size, num_feats)

        Returns:
            (batch_size, num_points, 1)
        """
        # TODO: optimize: try not to use permute here.

        batch_size, num_points, _ = query_points.shape

        x = self.mlp1(query_points.permute(0, 2, 1))
        x = torch.cat(
            (x, global_features.permute(0, 2, 1)),
            axis=1)
        x = self.mlp2(x)
        x = self.mlp3(x)

        return x.permute(0, 2, 1)


class SDFLocalDecoder(nn.Module):

    def __init__(self, out_features, batch_norm=False):
        super().__init__()

        self.mlp1 = make_mlp([3, 64, 256, 512], batch_norm=batch_norm)
        # TODO: Get the actual dimension of the feats
        self.mlp2 = make_mlp([1984, 512, 256], batch_norm=batch_norm)
        self.mlp3 = make_mlp([256, out_features], activation=None,
                             batch_norm=False)

    def forward(self, query_points, local_features):
        """
        Args:
            query_points: (batch_size, num_points, 3)
            local_features: (batch_size, num_points, num_feats)

        Returns:
            (batch_size, num_points, 1)
        """
        # TODO: optimize: try not to use permute here.

        batch_size, num_points, _ = query_points.shape

        x = self.mlp1(query_points.permute(0, 2, 1))
        x = torch.cat((x, local_features.permute(0, 2, 1)), axis=1)
        x = self.mlp2(x)
        x = self.mlp3(x)

        return x.permute(0, 2, 1)


class DISNEncoder(nn.Module):

    def __init__(self,
                 image_size=None,
                 use_pretrained_image_encoder=True,
                 local_feature_size=137,
                 image_encoding_dim=1000,
                 normalize=True,
                 resize_local_feature=True,
                 resize_input_shape=True,
                 in_channels=3):

        super().__init__()

        self.image_size = image_size
        self.local_feature_size = local_feature_size
        self.use_pretrained_image_encoder = use_pretrained_image_encoder
        self.resize_local_feature = resize_local_feature
        self.resize_input_shape = resize_input_shape
        self.image_encoder = VGG16WithFeatures(
            pretrained=use_pretrained_image_encoder,
            num_classes=image_encoding_dim,
            in_channels=in_channels)
        self.normalize = normalize

    def forward(self, images):
        """
        Args:
            images: (batch_size, channels, width, height). value in range [0, 1]

        Returns:
            Tuple: (global_features, encoder_features)
            global_features: the feature for each image (batch_size, num_feats).
            encoder_features: a tuple of output from the image encoder's intermediate layers.
        """
        # resize image

        if self.resize_input_shape and self.image_size is not None and (
                images.shape[2] != self.image_size or
                images.shape[3] != self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear')

        # encode image
        if self.normalize:
            # import ipdb
            # ipdb.set_trace()
            images = normalize_imagenet(images)
        # adding forward hook to obtain intermediate features from encoder
        # if self.normalize:

        global_features, encoder_outputs = self.image_encoder(images)
        if self.resize_local_feature:
            resized_outputs = [
                F.interpolate(
                    output,
                    size=(self.local_feature_size, self.local_feature_size),
                    mode='bilinear')
                for output in encoder_outputs
            ]
        else:
            resized_outputs = encoder_outputs
            # encoder_features = torch.cat(resized_outputs, axis=1)
        # encoder_features = resized_outputs
        # print('==> Length of feature ', len(resized_outputs))
        resized_outputs.insert(0, global_features)
        return resized_outputs


class DISNDecoder(nn.Module):
    def __init__(self,
                 out_features,
                 batch_norm=False):
        super().__init__()

        self.sdf_global_decoder = SDFGlobalDecoder(out_features,
                                                   batch_norm=batch_norm)
        self.sdf_local_decoder = SDFLocalDecoder(out_features,
                                                 batch_norm=batch_norm)

    def _project_points_to_image(self, points, camera_matrix):
        """
        Project points using the matrix.
        Args:
            points: Query points. (batch_size, num_points, 3)
            camera_matrix: Transform from world space to screen space.
                (batch_size, 4, 4).

        Returns:
            Projected points. (batch_size, num_points, 2)
        """
        batch_size, num_points, _ = points.shape
        homogeneous_points = torch.cat(
            (points,
             torch.ones(batch_size, num_points, 1, device=points.device)),
            axis=2)
        projected_points = torch.matmul(homogeneous_points, camera_matrix)
        # This is because of how DVR dataset works.
        projected_points = projected_points / projected_points[:, :, 2:3]
        return projected_points[:, :, :2]

    def _extract_point_image_features(self, encoder_features, query_points,
                                      camera_matrix):
        """
        Extract local image for each query point.
        Args:
            encoder_features: List of intermediate output from image encoder.
                TODO shape
            query_points: Query points. (batch_size, num_points, 3)
            camera_matrix: Transform from world space to screen space.
                (batch_size, 4, 4).

        Returns:
            The local image features at each query point.
            (batch_size, num_points, num_feats)
        """
        import ipdb
        ipdb.set_trace()
        projected_points = self._project_points_to_image(
            query_points, camera_matrix)
        # (B, num_points, 2), in [-1, 1], where (-1, -1) is top left and (1, 1)
        # is bottom right.

        sampled_features = F.grid_sample(
            encoder_features, projected_points.unsqueeze(2),
            mode='bilinear',
            padding_mode='zeros').squeeze(3)  # (B, num_feats, num_points)

        return sampled_features.permute(0, 2, 1)  # (B, num_points, num_feats)

    def forward(self, encoded_features,
                camera_matrix=None, tmp=None):
        """
        Args:
            query_points: Query points in world space.
                (batch_size, num_points, 3).
            encoded_features: Output from DISNEncoder.
            camera_matrix: Transform from world space to screen space.
                (batch_size, 4, 4).
                Each matrix A in batch should be such that xA where x is a
                row vector would project the point x to screen space, and that
                (-1, -1) is top left of the screen and (1, 1) is bottom right.
                Note that this is the transpose of a "traditional" projection
                matrix.

        Returns:
            SDF value for each query points. TODO shape
        """
        # import ipdb
        # ipdb.set_trace()
        # import ipdb
        # ipdb.set_trace()
        global_features = encoded_features[:, :, :1000]
        local_features = encoded_features[:, :, 1000:-3]
        query_points = encoded_features[:, :, -3:]
        # global_features, local_features, query_points = encoded_features

        # local_features = self._extract_point_image_features(
        #     encoder_features, query_points, camera_matrix)

        # decode SDF

        global_pred = self.sdf_global_decoder(query_points, global_features)
        local_pred = self.sdf_local_decoder(query_points, local_features)
        pred = global_pred + local_pred

        return pred.permute(0, 2, 1)
