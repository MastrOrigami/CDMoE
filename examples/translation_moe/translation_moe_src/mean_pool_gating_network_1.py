# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from cuml.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias = gate_bias)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        gate_score = F.softmax(gate_top_k_val, dim=-1)
        self.set_loss(torch.zeros(1, requires_grad=True).to(inp.device))

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class FusionNetwork(nn.Module):
    def __init__(self, feature_size):
        super(FusionNetwork, self).__init__()
        self.weight_matrix = nn.Parameter(torch.randn(feature_size, feature_size))

    def forward(self, output1, output2):
        fusion_output = torch.matmul(output1, self.weight_matrix) + torch.matmul(output2, self.weight_matrix)
        return fusion_output


class Correction(nn.Module):
    def __init__(self, input_size, output_size):
        super(Correction, self).__init__()

    def forward(self, x):
        return 0


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        hiden_size = 512
        self.batch_norm = nn.BatchNorm1d(hiden_size)
        self.fc1 = nn.Linear(input_size, hiden_size)
        self.fc2 = nn.Linear(hiden_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.batch_norm(self.fc1(x))))


class ClusterSelectModule(nn.Module):
    def __init__(self, input_dim, num_classes, k, num_clusters):
        super(ClusterSelectModule, self).__init__()
        self.k = k
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=6, random_state=42, init='k-means++', n_init=10, max_iter=30)

    def forward(self, features):
        features_np = features.detach().cpu().numpy().astype(float)
        cluster_assignments = self.kmeans.fit_predict(features_np)
        return {
            "cluster_assignments": cluster_assignments,
        }


class MeanPoolGatingNetwork1(NaiveGate):
    def __init__(self, d_model, num_expert, dropout=None):
        super().__init__(d_model, num_expert, world_size=1, top_k=1, gate_bias=True)
        self.embed_dim = d_model
        self.num_experts = 6
        self.counts = torch.zeros(num_expert, dtype=torch.int32).to("cuda")
        self.networks = nn.ModuleList([SimpleNN(d_model, 1) for _ in range(num_expert)])

        self.cluster_module = ClusterSelectModule(input_dim=d_model, num_classes=num_expert, k=1,
                                                  num_clusters=num_expert)

    def expert_vote(self, inp):
        outputs = []
        for network in self.networks:
            outputs.append(network(inp))
        return outputs

    def forward(self, encoder_out):
        if not (
                "encoder_out" in encoder_out
                and "encoder_padding_mask" in encoder_out
                and encoder_out["encoder_out"][0].size(2) == self.embed_dim
        ):
            raise ValueError("Unexpected format for encoder_out")

        encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
        encoder_out = encoder_out["encoder_out"][0].transpose(0, 1)
        if encoder_padding_mask is not None:
            encoder_out = encoder_out.clone()
            encoder_out[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(encoder_out, dim=1) / ntokens.type_as(encoder_out)
        else:
            x = torch.mean(encoder_out, dim=1)
        cluster_output = self.cluster_module(x)
        gate = torch.stack(self.expert_vote(x)).squeeze(-1).permute(1, 0)
        one_hot_encoded = torch.tensor(np.eye(self.num_experts)[cluster_output["cluster_assignments"]],
                                       dtype=torch.float).cuda()
        gate = gate * one_hot_encoded
        return F.log_softmax(gate, dim=-1)


class MeanPoolGatingNetwork(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.

    This module applies mean pooling over an encoder's output and returns
    reponsibilities for each expert. The encoder format is expected to match
    :class:`fairseq.models.transformer.TransformerEncoder`.
    """

    def __init__(self, embed_dim, num_experts, dropout=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = 6

        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = torch.nn.Linear(embed_dim, num_experts)

    def forward(self, encoder_out):
        if not (
                "encoder_out" in encoder_out
                and "encoder_padding_mask" in encoder_out
                and encoder_out["encoder_out"][0].size(2) == self.embed_dim
        ):
            raise ValueError("Unexpected format for encoder_out")

        encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
        encoder_out = encoder_out["encoder_out"][0].transpose(0, 1)
        if encoder_padding_mask is not None:
            encoder_out = encoder_out.clone()
            encoder_out[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(encoder_out, dim=1) / ntokens.type_as(encoder_out)
        else:
            x = torch.mean(encoder_out, dim=1)

        x = torch.tanh(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)
