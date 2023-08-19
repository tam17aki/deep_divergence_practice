# -*- coding: utf-8 -*-
"""Provides model definition.

Copyright (C) 2023 by Akira TAMAMORI
Copyright (C) 2023 by Fred Lu, Edward Raff, Francis Ferraro

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
"""
import torch
from omegaconf import DictConfig
from pytorch_metric_learning.distances import BaseDistance
from torch import linalg, nn


class Embedding(nn.Module):
    """Embedding network."""

    def __init__(self, cfg: DictConfig):
        """Initialize class."""
        super().__init__()
        input_dim = cfg.model.embedding.input_dim
        hidden_dim = cfg.model.embedding.hidden_dim  # list
        latent_dim = cfg.model.embedding.latent_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),  # 2 -> 1000
            nn.LayerNorm(hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),  # 1000 -> 500
            nn.LayerNorm(hidden_dim[1]),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim[1], latent_dim)
        self.cfg = cfg

    def forward(self, inputs):
        """Perform forward propagation.

        Args:
            inputs (Tensor) : input

        Returns:
            outputs (Tensor) : embedding
        """
        hidden = self.layers(inputs)
        outputs = self.fc_out(hidden)
        return outputs

    @torch.no_grad()
    def get_embeddings(self, dataset):
        """Return embeddings.

        This method is intended to run in inference mode.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.inference.n_batch,
            shuffle=False,
            drop_last=False,
        )
        begin = 0
        end = 0
        for i, batch in enumerate(dataloader):
            data, label = batch
            data, label = data.to(device), label.to(device)
            data = data.float()
            label = label.long()
            if self.cfg.model.euc_dist is False:  # use moment matching
                # compute mean over points on empirical dist.
                data = data.mean(dim=1)  # this comes from moment matching div.
            embed = self.forward(data)
            if i == 0:
                labels = torch.zeros(len(dataset), 1, device=device, dtype=label.dtype)
                embeddings = torch.zeros(
                    len(dataset), embed.size(1), device=device, dtype=data.dtype
                )
            end = begin + embed.size(0)
            embeddings[begin:end] = embed
            labels[begin:end] = label.unsqueeze(1)
            begin = end
        return embeddings, labels.squeeze(1)


class EuclideanDistance(BaseDistance):
    """Euclidean Distance.

    This implementaion is a fork of the source codes provied by the authors of
    'Neural Bregman Divergences for Distance Learning,' ICLR 2023.
    https://openreview.net/forum?id=nJ3Vx78Nf7p
    """

    def __init__(self, squared=False, **kwargs):
        """Initialize class."""
        super().__init__(**kwargs)
        self.squared = squared
        if squared:
            self.post_fn = lambda x: torch.pow(x, 2)
        else:
            self.post_fn = lambda x: x

    def compute_mat(self, query_emb, ref_emb):
        """Compute distance matrix."""
        query3d = query_emb[:, None, :]  # [N, 1, D]
        ref3d = ref_emb[None, :, :]  # [1, N, D]
        mat = linalg.vector_norm(query3d - ref3d, dim=-1)  # [N, N]
        return self.post_fn(mat)  # [N, N]

    def pairwise_distance(self, query_emb, ref_emb):
        """Compute pairwise distance."""
        pdist = linalg.norm(query_emb - ref_emb, dim=-1)  # [N]
        return self.post_fn(pdist)


class MomentMatching(BaseDistance):
    """Moment Matching divergence."""

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def compute_mat(self, query_emb, ref_emb):
        """Compute distance matrix."""
        n_batch = query_emb.shape[0]
        points_q = torch.reshape(
            query_emb, (n_batch, -1, self.cfg.model.embedding.latent_dim)
        )  # [N, S, 2]; S=#samples
        points_r = torch.reshape(
            ref_emb, (n_batch, -1, self.cfg.model.embedding.latent_dim)
        )  # [N, S, 2]
        mean_q = torch.mean(points_q, dim=1)  # [N, 2] -> mean over points
        mean_r = torch.mean(points_r, dim=1)  # [N, 2] -> mean over points
        query3d = mean_q[:, None, :]  # [N, 1, 2]
        ref3d = mean_r[None, :, :]  # [1, N, 2]
        mat = linalg.vector_norm(query3d - ref3d, dim=-1)  # [N, N]
        return torch.pow(mat, 2)

    def pairwise_distance(self, query_emb, ref_emb):
        """Compute pairwise distance."""
        n_batch = query_emb.shape[0]
        points_q = torch.reshape(
            query_emb, (n_batch, -1, self.cfg.model.embedding.latent_dim)
        )  # [N, S, 2]; S=#samples
        points_r = torch.reshape(
            ref_emb, (n_batch, -1, self.cfg.model.embedding.latent_dim)
        )  # [N, S, 2]
        mean_q = torch.mean(points_q, dim=1)  # [N, 2] -> mean over points
        mean_r = torch.mean(points_r, dim=1)  # [N, 2] -> mean over points
        pdist = linalg.norm(mean_q - mean_r, dim=-1)  # [N]
        return torch.pow(pdist, 2)


def get_model(cfg, device):
    """Instantiate metric and embedding functions."""
    if cfg.model.euc_dist:
        metric = EuclideanDistance(
            squared=cfg.model.euc_squared, normalize_embeddings=cfg.model.emb_normalize
        ).to(device)
    else:
        metric = MomentMatching(
            cfg=cfg, normalize_embeddings=cfg.model.emb_normalize
        ).to(device)
    embedding = Embedding(cfg).to(device)
    return metric, embedding
