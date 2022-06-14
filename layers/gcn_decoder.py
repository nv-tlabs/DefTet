'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.matrix_utils import sparse_batch_matmul
from layers.pv_utils import create_mlp_components

def normalize_sparse_matrix(matrix):
    indices = matrix._indices().clone().cpu()
    values = matrix._values().clone().cpu()

    degrees = {}
    for i in range(indices.shape[1]):
        node = float(indices[0, i])
        degrees[node] = degrees.get(node, 0) + 1

    for i in range(indices.shape[1]):
        values[i] /= degrees[float(indices[0, i])]

    return torch.sparse.FloatTensor(
        indices, values, matrix.shape).to(matrix.device)


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.filter = nn.Linear(input_dim, output_dim)
        self.self_filter = nn.Linear(input_dim, output_dim)
        # self.initialize()

    def initialize(self):
        for f in [self.filter, self.self_filter]:
            nn.init.xavier_uniform_(f.weight.data)
            if f.bias is not None:
                f.bias.data.zero_()

    def forward(self, node_feat, normalized_adj, use_sparse=False):
        '''
        Args:
            node_feat: features on each node (batch_size, num_nodes, input_dim)
            normalized_adj: normalized adjacency matrix (num_nodes, num_nodes)

        Returns:
            new_node_feat: (batch_size, num_nodes, output_dim)
        '''
        if not use_sparse:
            return self.filter(node_feat) + self.filter(torch.bmm(normalized_adj, node_feat))
        return self.self_filter(node_feat) + \
            self.filter(sparse_batch_matmul(normalized_adj, node_feat))

    def reset_parameters(self):
        self.filter.weight.data = self.filter.weight.data * 0.1
        # self.filter.bias.data = self.filter.bias.data * 0.1
        self.self_filter.weight.data = self.self_filter.weight.data * 0.1
        # self.self_filter.bias.data = self.self_filter.bias.data * 0.1

class GraphConvLayer(nn.Module):

    def __init__(self, size_in, size_out=None,  c_dim=128, use_attention=False, norm_method='batch_norm',
                 use_c_bn=True):
        super(GraphConvLayer, self).__init__()

        # Attributes
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_out = size_out
        self.use_c_bn = use_c_bn
        self.conv = GraphConv(size_in, size_out)

        self.activation = nn.ReLU()

    def forward(self, x, normalized_adj, c=None, use_sparse=False):
        if self.use_c_bn:
            x = self.bn(x.transpose(1, 2), c).transpose(1, 2)
        result = self.conv(self.activation(x), normalized_adj, use_sparse=use_sparse)
        return result

    def reset_parameters(self):
        self.conv.reset_parameters()

class GraphConvBlock(nn.Module):

    def __init__(self, size_in, c_dim=128, use_c_bn=False, size_h=None, size_out=None, norm_method='batch_norm', use_attention=False):
        super(GraphConvBlock, self).__init__()

        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.use_c_bn = use_c_bn

        # Submodules
        self.layer_0 = GraphConvLayer(
            c_dim=c_dim, size_in=size_in, size_out=size_h, use_c_bn=use_c_bn, norm_method=norm_method, use_attention=use_attention)
        self.layer_1 = GraphConvLayer(
            c_dim=c_dim, size_in=size_h, size_out=size_out, use_c_bn=use_c_bn, norm_method=norm_method, use_attention=use_attention)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out)

        # Initialization
        # nn.init.zeros_(self.conv_1.weight)

    def forward(self, x,  normalized_adj, c=None, use_sparse=True):
        net = self.layer_0(x,  normalized_adj, c, use_sparse=use_sparse)
        dx = self.layer_1(net,  normalized_adj, c, use_sparse=use_sparse)

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class GCNDecoder(nn.Module):
    """
    Based on: LanczosNetwork/gcn.py at master · lrjconan/LanczosNetwork:
    https://github.com/lrjconan/LanczosNetwork/blob/master/model/gcn.py

    Graph Convolutional Networks,
    see reference below for more information
    Kipf, T.N. and Welling, M., 2016.
    Semi-supervised classification with graph convolutional networks.
    arXiv preprint arXiv:1609.02907.
    """

    def __init__(self,
                 c_dim,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 adj_sparse=None,
                 use_attention=False,
                 require_latent=False,
                 use_learned_def_mask=False,
                 use_c_bn=True):
        '''
        adj_sparse: adjacency matrix (batch_size, num_nodes, num_nodes) that contains self-loop.
        '''

        super(GCNDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # assert adj_sparse is not None, 'Adjacency matrix is not given'
        if adj_sparse:
            self.normalized_adj = normalize_sparse_matrix(adj_sparse)
            self.normalized_adjs = {}

        # a = self.normalized_adj.to_dense().sum(dim=1)
        # print(a)

        self.use_learned_def_mask = use_learned_def_mask
        self.learned_def_mask = nn.Parameter(
            torch.zeros(output_dim)) if self.use_learned_def_mask else None

        self.require_latent = require_latent
        if require_latent and not self.hidden_dims:
            raise ValueError(
                'There must be at least one hidden layer when require_latent is True')

        dim_list = [self.input_dim] + self.hidden_dims + [self.output_dim]

        self.initial_layer = nn.Linear(dim_list[0], dim_list[1])

        layers = []

        for i in range(1, len(self.hidden_dims)):
            layers.append(GraphConvBlock(
                c_dim=c_dim, size_in=dim_list[i], size_out=dim_list[i + 1], use_attention=use_attention))

        layers.append(GraphConvLayer(
            c_dim=c_dim, size_in=dim_list[-2], size_out=dim_list[-1], use_attention=use_attention, use_c_bn=use_c_bn))

        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

    def get_normalized_adj(self, device):
        if device not in self.normalized_adjs:
            self.normalized_adjs[device] = self.normalized_adj.to(device)

        return self.normalized_adjs[device]

    def get_latent_dims(self):
        if not self.require_latent:
            raise ValueError(
                'Cannot get latent dims when require_latent is False')

        return self.hidden_dims[-1]

    def forward(self, p, z=None, c=None, adj=None, use_sparse=True):
        """
        Args:
            p: features on each node (batch_size, num_nodes, input_dim)
            z: unused
            c: class conditional features

        Returns:
            new_node_feat: (batch_size, num_nodes, output_dim)
        """
        p = self.initial_layer(p)
        for i in range(len(self.layers)):
            layer = self.layers[i]

            if self.require_latent and i == len(self.layers) - 1:
                latent = p
            if not (adj is None):
                p = layer(p, c=c, normalized_adj=adj, use_sparse=False)
            else:
                p = layer(p, c=c, normalized_adj=self.get_normalized_adj(p.device))

        if self.use_learned_def_mask:
            p *= self.learned_def_mask

        if self.require_latent:
            return p, latent

        else:
            return p

    def reset_parameters(self):
        self.layers[-1].reset_parameters()


class GCNMLPDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 gcn_hidden_dims,
                 mlp_hidden_dims,
                 output_dim,
                 adj_sparse,
                 c_dim=128,
                 use_attention=False,
                 require_latent=False,
                 use_learned_def_mask=False):
        '''
        adj_sparse: adjacency matrix (batch_size, num_nodes, num_nodes) that contains self-loop.
        '''

        super(GCNMLPDecoder, self).__init__()
        self.input_dim = input_dim
        self.gcn_hidden_dims = gcn_hidden_dims
        self.mlp_hidden_dims = mlp_hidden_dims
        self.output_dim = output_dim
        # import ipdb
        # ipdb.set_trace()
        assert not use_learned_def_mask
        assert not use_attention
        assert adj_sparse is not None, 'Adjacency matrix is not given'
        self.normalized_adj = adj_sparse
        self.normalized_adjs = {}

        self.use_learned_def_mask = use_learned_def_mask

        self.require_latent = require_latent

        dim_list = [self.input_dim] + self.gcn_hidden_dims + self.mlp_hidden_dims + [self.output_dim]

        self.initial_layer = nn.Linear(dim_list[0], dim_list[1])

        gcn_layers = []

        for i in range(0, len(self.gcn_hidden_dims) - 1):
            gcn_layers.append(GraphConvBlock(
                c_dim=c_dim, size_in=self.gcn_hidden_dims[i], size_out=self.gcn_hidden_dims[i + 1], use_attention=use_attention, use_c_bn=False))


        self.gcn_layers = nn.ModuleList(gcn_layers)
        mlp_layers, _ = create_mlp_components(in_channels=self.gcn_hidden_dims[-1],
                                          out_channels=mlp_hidden_dims,
                                          classifier=True, dim=2, width_multiplier=1)

        self.mlp_layers = nn.Sequential(*mlp_layers)


    def get_normalized_adj(self, device):
        if device not in self.normalized_adjs:

            self.normalized_adjs[device] = self.normalized_adj.construct().to(device)

        return self.normalized_adjs[device]

    def forward(self, p):
        """
        Args:
            p: features on each node (batch_size, num_nodes, input_dim)
            z: unused
            c: class conditional features

        Returns:
            new_node_feat: (batch_size, num_nodes, output_dim)
        """
        # import ipdb
        # ipdb.set_trace()
        p = self.initial_layer(p.transpose(1, 2))
        adj = self.get_normalized_adj(p.device)
        for i in range(len(self.gcn_layers)):
            layer = self.gcn_layers[i]
            p = layer(p, adj)
            if self.require_latent and i == len(self.gcn_layers) - 1:
                latent = p
                latent = latent.permute(0, 2, 1)
        # import ipdb
        # ipdb.set_trace()
        p = p.transpose(1, 2)
        p = self.mlp_layers(p)

        if self.require_latent:
            return p, latent
        else:
            return p
