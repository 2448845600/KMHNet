import os
import pickle
from logging import getLogger

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import linalg

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

import pickle

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class PathAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(PathAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

    def forward(self, feat, padding_mask):
        """
        Args:
            feat: (L, N, E) where L is the target sequence length, N is the batch size, E is the embedding dimension
            padding_mask: (N, S)` where N is the batch size, S is the source sequence length.
        """
        attn_output, attn_output_weights = self.attention(feat, feat, feat, key_padding_mask=padding_mask)
        # feat = feat + attn_output
        return attn_output


class LinearBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(c_in, c_out, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MHScoreLayer(nn.Module):
    def __init__(self, max_path_len, input_dim, node_num, pool_layer=nn.MaxPool1d):
        super(MHScoreLayer, self).__init__()
        self.max_path_len = max_path_len
        self.input_dim = input_dim
        self.node_num = node_num
        self.hop_max_pooling = pool_layer(kernel_size=self.max_path_len)
        self.fcb = nn.Sequential(
            LinearBlock(self.input_dim, self.input_dim // 2),
            LinearBlock(self.input_dim // 2, self.input_dim // 2)
        )
        self.fc = nn.Linear(self.input_dim // 2, 1)

    def forward(self, feat, padding_mask, fusion_type='maxpooling'):
        """

        Args:
            feat: [L, B, D]
            padding_mask: [B, L]
            fusion_type:
        """
        feat = feat.permute(1, 2, 0)  # B, D, L
        padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, self.input_dim).permute(0, 2, 1)  # B, D, L
        # valued_mask = 1 - padding_mask
        # weight_mask = feat.sum(1).unsequence(1).repeat(1, self.input_dim, 1)  # B, D, L
        feat = feat.masked_fill(padding_mask, float('-inf'))

        if fusion_type == 'maxpooling':
            feat = self.hop_max_pooling(feat).squeeze(-1)
        # elif fusion_type == 'attention':
        #     # 将 attention 的输出，加权，
        #     feat = feat * weight_mask


        feat = self.fcb(feat)
        score = self.fc(feat)
        return score


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class MHopGWNET(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 2)
        super().__init__(config, data_feature)

        self.dataset = config.get('dataset', 'SZBus')
        self.dropout = config.get('dropout', 0.3)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)
        self.gcn_bool = config.get('gcn_bool', True)
        self.use_rel_adj = config.get('use_rel_adj', True)
        self.use_apt_adj = config.get('use_apt_adj', True)
        self.adjtype = config.get('adjtype', 'doubletransition')
        self.randomadj = config.get('randomadj', True)
        self.kernel_size = config.get('kernel_size', 2)
        self.nhid = config.get('nhid', 32)
        self.residual_channels = config.get('residual_channels', self.nhid)
        self.dilation_channels = config.get('dilation_channels', self.nhid)
        self.skip_channels = config.get('skip_channels', self.nhid * 8)
        self.end_channels = config.get('end_channels', self.nhid * 16)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        # init multi-hop setting
        self.ke_model = config.get('ke_model', 'TransE')
        self.ke_dim = config.get('ke_dim', 200)
        self.max_hop = config.get('max_hop', 2)
        self.use_rl = config.get('use_rl', True)
        self.use_inner_node = config.get('use_node', True)
        self.feat_fuse = config.get('feat_fuse', 'cat')
        self.path_fuse = config.get('path_fuse', 'maxpooling')

        self.mh_encoder_type = config.get('mh_encoder_type', 'Linear')
        self.attn_num_layers = config.get('attn_num_layers', 1)
        self.attn_num_heads = config.get('attn_num_heads', 1)

        # process path feat
        self.path_feat_path = 'kg_utils/path_feature/{}/{}_{}_{}hop.pkl'.format(
            self.dataset, self.ke_model, self.ke_dim, self.max_hop)
        with open(self.path_feat_path, mode='rb') as f:
            origin_path_feats = pickle.load(f)
        self._logger.info('load path_feat from {}'.format(self.path_feat_path))
        max_path_len, self.mh_feat, self.mh_value_mask, self.mh_padding_mask, self.i1d_to_ij2d = \
            self.process_feats(origin_path_feats)  # 注意，value_mask与padding_mask的意义相反

        # init mh_layers
        self.mh_feat_encoder = None
        # if self.mh_encoder_type == 'Attention':
        attn_layers = [PathAttention(embedding_dim=self.ke_dim, num_heads=self.attn_num_heads) for _ in range(self.attn_num_layers)]
        self.mh_feat_encoder = nn.Sequential(*attn_layers)
        print(self.mh_feat_encoder)
        self.mh_score_layer = MHScoreLayer(max_path_len, self.ke_dim, self.num_nodes)

        self.apt_layer = config.get('apt_layer', True)
        if self.apt_layer:
            self.layers = np.int(
                np.round(np.log((((self.input_window - 1) / (self.blocks * (self.kernel_size - 1))) + 1)) / np.log(2)))
            print('# of layers change to %s' % self.layers)

        self._scaler = self.data_feature.get('scaler')

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        # init rel, poi, apt types of supports
        self.supports = []
        if self.use_rel_adj:
            self.cal_adj(self.adjtype)
            self.supports += [torch.tensor(i).to(self.device) for i in self.adj_mx]
            self._logger.info('add adj_mx to supports')

        self.supports_len = len(self.supports)

        self.aptinit = None if self.randomadj else self.supports[0]
        if self.gcn_bool and self.use_apt_adj:
            if self.aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                             requires_grad=True).to(self.device)
            else:
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
            self.supports_len += 1
            self._logger.info('add apt_adj(v1 * v2) in forward process')

        if self.max_hop > 0:
            self.supports_len += 1
            self._logger.info('add multi hop score in forward process')

        # init layers
        receptive_field = self.output_dim
        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: ' + str(self.receptive_field))

        # if os.path.exists('cache.csv'):
        #     mh_score = np.loadtxt('cache.csv')
        #     mh_score = torch.from_numpy(mh_score).to(self.device)
        # else:
        #     if self.mh_feat_encoder is not None:
        #         for att_layer in self.mh_feat_encoder:
        #             att_out = att_layer(self.mh_feat, self.mh_padding_mask)
        #             self.mh_feat = self.mh_feat + att_out
        #     score = self.mh_score_layer(self.mh_feat, self.mh_padding_mask)
        #     mh_score = torch.zeros((self.num_nodes, self.num_nodes)).to(self.device)
        #     for i1d, (i2d, j2d) in self.i1d_to_ij2d.items():
        #         mh_score[i2d][j2d] = score[i1d]
        #     np.savetxt('cache.csv', mh_score.tensor.detach().numpy())
        #     self.eval_mh_score = mh_score

    def process_feats(self, origin_path_feats):
        """
        returns:
            mh_feat: (L, NxN, D)
            mh_value_mask: (N, N) -> 1 表示有意义的位置
            mh_padding_mask: (NxN, L) -> 1 表示被 padding 的位置
            max_path_len: int
        """
        max_path_len = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if origin_path_feats[i][j] is not None:
                    max_path_len = max(max_path_len, origin_path_feats[i][j].shape[1] // self.ke_dim)

        padding_feat, padding_mask, i1d_to_ij2d, value_mask = [], [], {}, torch.zeros((self.num_nodes, self.num_nodes))
        cur_i1d = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if origin_path_feats[i][j] is not None:
                    zero_feat = torch.zeros((1, max_path_len, self.ke_dim))
                    one_mask = torch.zeros(1, max_path_len)
                    value_feat = origin_path_feats[i][j].view(-1, self.ke_dim)
                    zero_feat[:, :value_feat.shape[0]] = value_feat
                    one_mask[:, :value_feat.shape[0]] = False  # 将没有被 padding 的位置置为 0
                    padding_feat.append(zero_feat)
                    padding_mask.append(one_mask)
                    value_mask[i][j] = True  # 将没有被 padding 的位置置为 1
                    i1d_to_ij2d[cur_i1d] = [i, j]
                    cur_i1d += 1

        mh_feat = torch.cat(padding_feat, dim=0).permute(1, 0, 2).to(self.device)
        mh_padding_mask = torch.cat(padding_mask, dim=0).to(self.device).bool()
        mh_value_mask = value_mask.to(self.device).bool()
        return max_path_len, mh_feat, mh_value_mask, mh_padding_mask, i1d_to_ij2d

    def forward(self, batch):
        inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        mh_feat = self.mh_feat.detach()
        mh_padding_mask = self.mh_padding_mask.detach()

        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.use_apt_adj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        #####################################################################################
        if self.mh_feat_encoder is not None:
            for att_layer in self.mh_feat_encoder:
                att_out = att_layer(mh_feat, mh_padding_mask)
                mh_feat = mh_feat + att_out
        score = self.mh_score_layer(mh_feat, mh_padding_mask)
        mh_score = torch.zeros((self.num_nodes, self.num_nodes)).to(self.device)
        for i1d, (i2d, j2d) in self.i1d_to_ij2d.items():
            mh_score[i2d][j2d] = score[i1d]

        np.savetxt('MHop_Correlation.csv', mh_score.detach().cpu().numpy())
        exit()
        new_supports += [mh_score]

        # new_supports += [self.eval_mh_score]

        #####################################################################################

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.use_apt_adj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        x = F.relu(skip)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        # (batch_size, output_window, num_nodes, self.output_dim)
        return x

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # print('y_true', y_true.shape)
        # print('y_predicted', y_predicted.shape)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
