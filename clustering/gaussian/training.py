# -*- coding: utf-8 -*-
"""Training script.

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
from collections import namedtuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from progressbar import progressbar as prg

from dataset import get_dataset
from factory import get_loss_miner, get_optimizer
from model import get_model
from util import (append_stats, calc_accuracy, get_device, init_manual_seed,
                  init_stats, print_stats, save_checkpoint)


def get_training_modules(cfg: DictConfig, device):
    """Instantiate modules for training."""
    embedding, metric = get_model(cfg, device)
    optimizer = get_optimizer(cfg, embedding, metric)
    loss_func, mining_func = get_loss_miner(cfg, metric)
    TrainingModules = namedtuple(
        "TrainingModules", ["embedding", "optimizer", "loss_func", "mining_func"]
    )
    modules = TrainingModules(embedding, optimizer, loss_func, mining_func)
    return modules


def training_loop(cfg: DictConfig, dataset, modules, device):
    """Perform training loop."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.n_batch, shuffle=True
    )
    embedding, optimizer, loss_func, mining_func = modules
    embedding.train()
    for _ in range(cfg.training.n_epoch):
        for (points, labels) in dataloader:
            points, labels = points.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            embeds = embedding(points)
            if cfg.model.euc_dist is False:  # to pass loss and mining func.
                embeds = embeds.reshape(
                    -1, cfg.training.n_points * cfg.model.embedding.latent_dim
                )
            loss = loss_func(embeds, indices_tuple=mining_func(embeds, labels))
            loss.backward()
            optimizer.step()


def main(cfg: DictConfig):
    """Perform training and calculate metric accuracies."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration
    device = get_device()
    init_manual_seed(0)  # fix seed

    # perform training loops changing random seed for dataset
    all_stats = init_stats()
    for seed in prg(range(cfg.training.n_trial)):
        train_dataset, test_dataset = get_dataset(cfg, seed)
        modules = get_training_modules(cfg, device)  # instantiate modules for training
        training_loop(cfg, train_dataset, modules, device)
        append_stats(
            all_stats,
            calc_accuracy(cfg, train_dataset, test_dataset, modules.embedding, seed),
        )
        save_checkpoint(cfg, modules.embedding, seed)

    print_stats(all_stats)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
