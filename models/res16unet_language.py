###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2024
###########################################################################

import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr
from models.modules.resnet_block import BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


class Res16UNetBaseLang(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self, in_channels, out_channels, config, D=3, out_fpn=False, **kwargs
    ):
        super().__init__(in_channels, out_channels, config, D)
        self.out_fpn = out_fpn

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn0 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum
        )

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn1 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum
        )
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum
        )
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum
        )
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn4 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum
        )
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr4 = get_norm(
            self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr5 = get_norm(
            self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr6 = get_norm(
            self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr7 = get_norm(
            self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.final = conv(
            self.PLANES[7],
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            D=D,
        )
        self.relu = MinkowskiReLU(inplace=True)
        self.res_gate_0 = nn.Sequential(
            nn.Linear(self.PLANES[3], self.PLANES[3], bias=False),
            nn.ReLU(),
            nn.Linear(self.PLANES[3], self.PLANES[3], bias=False),
            nn.Tanh()
        )
        self.fusion_0 = PWAM(self.PLANES[3],  # both the visual input and for combining, num of channels
                           self.PLANES[3],  # v_in
                           768,  # l_in
                           self.PLANES[3],  # key
                           self.PLANES[3],  # value
                           num_heads=1,
                           dropout=0.0,
                           input_channel=self.PLANES[3],
                           mlp=[self.PLANES[3], self.PLANES[3]]
                             )
        self.res_gate_1 = nn.Sequential(
            nn.Linear(self.PLANES[4], self.PLANES[4], bias=False),
            nn.ReLU(),
            nn.Linear(self.PLANES[4], self.PLANES[4], bias=False),
            nn.Tanh()
        )
        self.fusion_1 = PWAM(self.PLANES[4],  # both the visual input and for combining, num of channels
                           self.PLANES[4],  # v_in
                           768,  # l_in
                           self.PLANES[4],  # key
                           self.PLANES[4],  # value
                           num_heads=1,
                           dropout=0.0,
                             input_channel=self.PLANES[4],
                             mlp=[self.PLANES[4], self.PLANES[4]]
                             )
        self.res_gate_2 = nn.Sequential(
            nn.Linear(self.PLANES[5], self.PLANES[5], bias=False),
            nn.ReLU(),
            nn.Linear(self.PLANES[5], self.PLANES[5], bias=False),
            nn.Tanh()
        )
        self.fusion_2 = PWAM(self.PLANES[5],  # both the visual input and for combining, num of channels
                           self.PLANES[5],  # v_in
                           768,  # l_in
                           self.PLANES[5],  # key
                           self.PLANES[5],  # value
                           num_heads=1,
                           dropout=0.0,
                             input_channel=self.PLANES[5],
                             mlp=[self.PLANES[5], self.PLANES[5]]
                             )

        self.fusion_3 = PWAM(self.PLANES[6],  # both the visual input and for combining, num of channels
                           self.PLANES[6],  # v_in
                           768,  # l_in
                           self.PLANES[6],  # key
                           self.PLANES[6],  # value
                           num_heads=1,
                           dropout=0.0,
                             input_channel=self.PLANES[6],
                             mlp=[self.PLANES[6], self.PLANES[6]]
                             )
        self.res_gate_3 = nn.Sequential(
            nn.Linear(self.PLANES[6], self.PLANES[6], bias=False),
            nn.ReLU(),
            nn.Linear(self.PLANES[6], self.PLANES[6], bias=False),
            nn.Tanh()
        )
        self.fusion_4 = PWAM(self.PLANES[7],  # both the visual input and for combining, num of channels
                           self.PLANES[7],  # v_in
                           768,  # l_in
                           self.PLANES[7],  # key
                           self.PLANES[7],  # value
                           num_heads=1,
                           dropout=0.0,
                             input_channel=self.PLANES[7],
                             mlp=[self.PLANES[7], self.PLANES[7]]
                             )
        self.res_gate_4 = nn.Sequential(
            nn.Linear(self.PLANES[7], self.PLANES[7], bias=False),
            nn.ReLU(),
            nn.Linear(self.PLANES[7], self.PLANES[7], bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        feature_maps = []
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        vis_feat_list = []

        for vis_feat, xyz, lang_feat, lang_mask in zip(out.decomposed_features, out.decomposed_coordinates, l, l_mask):
            out_residual = self.fusion_0(vis_feat.unsqueeze(0), xyz.unsqueeze(0), lang_feat.unsqueeze(0).transpose(1, 2), lang_mask.unsqueeze(0))
            vis_feat = vis_feat + (self.res_gate_0(out_residual) * out_residual)
            vis_feat_list.append(vis_feat.squeeze(0))

        vis_feat_list = torch.cat(vis_feat_list)
        out = me.SparseTensor(
            features=vis_feat_list,
            coordinate_manager=out.coordinate_manager,
            coordinate_map_key=out.coordinate_map_key,
        )

        feature_maps.append(out)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        vis_feat_list = []
        for vis_feat, xyz, lang_feat, lang_mask in zip(out.decomposed_features, out.decomposed_coordinates, l, l_mask):
            out_residual = self.fusion_1(vis_feat.unsqueeze(0), xyz.unsqueeze(0), lang_feat.unsqueeze(0).transpose(1, 2), lang_mask.unsqueeze(0))
            vis_feat = vis_feat + (self.res_gate_1(out_residual) * out_residual)
            vis_feat_list.append(vis_feat.squeeze(0))
        vis_feat_list = torch.cat(vis_feat_list)
        out = me.SparseTensor(
            features=vis_feat_list,
            coordinate_manager=out.coordinate_manager,
            coordinate_map_key=out.coordinate_map_key,
        )
        feature_maps.append(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        vis_feat_list = []
        for vis_feat, xyz, lang_feat, lang_mask in zip(out.decomposed_features, out.decomposed_coordinates, l, l_mask):
            out_residual = self.fusion_2(vis_feat.unsqueeze(0), xyz.unsqueeze(0), lang_feat.unsqueeze(0).transpose(1, 2), lang_mask.unsqueeze(0))
            vis_feat = vis_feat + (self.res_gate_2(out_residual) * out_residual)

            vis_feat_list.append(vis_feat.squeeze(0))
        vis_feat_list = torch.cat(vis_feat_list)
        out = me.SparseTensor(
            features=vis_feat_list,
            coordinate_manager=out.coordinate_manager,
            coordinate_map_key=out.coordinate_map_key,
        )
        feature_maps.append(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        vis_feat_list = []
        for vis_feat, xyz, lang_feat, lang_mask in zip(out.decomposed_features, out.decomposed_coordinates, l, l_mask):
            out_residual = self.fusion_3(vis_feat.unsqueeze(0), xyz.unsqueeze(0), lang_feat.unsqueeze(0).transpose(1, 2), lang_mask.unsqueeze(0))
            vis_feat = vis_feat + (self.res_gate_3(out_residual) * out_residual)
            vis_feat_list.append(vis_feat.squeeze(0))
        vis_feat_list = torch.cat(vis_feat_list)

        out = me.SparseTensor(
            features=vis_feat_list,
            coordinate_manager=out.coordinate_manager,
            coordinate_map_key=out.coordinate_map_key,
        )

        feature_maps.append(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = me.cat(out, out_p1)
        out = self.block8(out)
        vis_feat_list = []
        for vis_feat, xyz, lang_feat, lang_mask in zip(out.decomposed_features, out.decomposed_coordinates, l, l_mask):
            out_residual = self.fusion_4(vis_feat.unsqueeze(0), xyz.unsqueeze(0), lang_feat.unsqueeze(0).transpose(1, 2), lang_mask.unsqueeze(0))
            vis_feat = vis_feat + (self.res_gate_4(out_residual) * out_residual)

            vis_feat_list.append(vis_feat.squeeze(0))
        vis_feat_list = torch.cat(vis_feat_list)
        out = me.SparseTensor(
            features=vis_feat_list,
            coordinate_manager=out.coordinate_manager,
            coordinate_map_key=out.coordinate_map_key,
        )

        feature_maps.append(out)

        if not self.out_fpn:
            return out
        else:
            return out, feature_maps

"""
    The architecture is based on LAVT.
    https://github.com/yz93/LAVT-RIS
"""

class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0, input_channel=256,  mlp= [256, 256]):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

        self.npoint, self.radius, self.nsample = 64, 0.5, 16
        last_channel = input_channel + 3
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def propagate(self, xyz1, xyz2, points1, points2, de_neighbors, pro_cof):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, N, D']
            points2: input points data, [B, S, D'']
        Return:
            new_points: upsampled points data, [B, N, D''']
        """

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :de_neighbors], idx[:, :, :de_neighbors]  # [B, N, S]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        weight = weight.view(B, N, de_neighbors, 1)

        interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)#B, N, 6, C->B,N,C
        new_points = interpolated_points
        return new_points

    def forward(self, x, xyz, l, l_mask):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        xyz = xyz / xyz.max()
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, x)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        B, C, self.nsample, self.npoint = new_points.shape
        new_points_ = new_points.reshape(B * self.npoint, C, self.nsample).transpose(1, 2)

        lang = self.image_lang_att(new_points_, l, l_mask)  # (B, H*W, dim)

        lang_propagate = self.propagate(xyz1=xyz, xyz2=new_xyz, points1=x, points2=lang, de_neighbors=3, pro_cof=0.1)

        mm = torch.mul(vis, lang_propagate.transpose(1, 2))
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(1, self.num_heads, self.key_channels//self.num_heads, n_l).repeat(B, 1, 1, 1)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(1, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = sim_map.sum(dim=2, keepdim=True).permute(1, 2, 0, 3)
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(1, -1, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out


class Res16UNet34(Res16UNetBaseLang):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet34CLang(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points