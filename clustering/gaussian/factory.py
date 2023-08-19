# -*- coding: utf-8 -*-
"""Provides optimizer, loss function, and mining function.

Copyright (C) 2023 by Akira TAMAMORI

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
import sys

from omegaconf import DictConfig
from pytorch_metric_learning import losses, miners, reducers
from torch import optim


def get_optimizer(cfg: DictConfig, embedding, metric):
    """Instantiate optimizer."""
    parameters = [*embedding.parameters(), *metric.parameters()]
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer = optimizer_class(parameters, **cfg.training.optim.optimizer.params)
    return optimizer


def get_loss_miner(cfg, distance):
    """Instantiate loss function and mining function."""
    if cfg.training.loss_type == "triplet":
        loss_name = cfg.training.triplet_loss.name
        loss_params = cfg.training.triplet_loss.params
    elif cfg.training.loss_type == "contrastive":
        loss_name = cfg.training.contrastive_loss.name
        loss_params = cfg.training.contrastive_loss.params
    else:
        sys.exit(
            "\nError: loss_type must be 'triplet' or 'contrastive'! "
            f"current value: {cfg.training.loss_type}"
        )

    if cfg.training.miner_type == "triplet":
        miner_name = cfg.training.triplet_miner.name
        miner_params = cfg.training.triplet_miner.params
    elif cfg.training.miner_type == "pair":
        miner_name = cfg.training.pair_miner.name
        miner_params = cfg.training.pair_miner.params
    else:
        sys.exit(
            "\nError: miner_type must be 'triplet' or 'pair'! "
            f"current value: {cfg.training.miner_type}"
        )

    losses_class = getattr(losses, loss_name)
    miners_class = getattr(miners, miner_name)
    reducer_class = getattr(reducers, cfg.training.reducers.name)
    loss_func = losses_class(
        **loss_params,
        distance=distance,
        reducer=reducer_class(**cfg.training.reducers.params),
    )
    mining_func = miners_class(**miner_params, distance=distance)
    return loss_func, mining_func
