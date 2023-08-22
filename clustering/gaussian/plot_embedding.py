# -*- coding: utf-8 -*-
"""Plot embedding result.

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
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from dataset import get_dataset
from model import get_model
from util import get_device, init_manual_seed

warnings.simplefilter("ignore", UserWarning)


def load_checkpoint(cfg: DictConfig, model):
    """Load checkpoint."""
    model_dir = os.path.join(cfg.directory.root_dir, cfg.directory.model_dir)
    if cfg.model.euc_dist is False:  # moment matching
        model_file = os.path.join(model_dir, cfg.training.model_file)
    else:  # Euclidean distance
        model_file = os.path.join(model_dir, cfg.training.model_euc_file)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)


def plot_result(train_dataset, test_dataset, model):
    """Plot clustering result."""
    embeds = {"train": None, "test": None}
    labels = {"train": None, "test": None}

    model.eval()
    embeds["train"], labels["train"] = model.get_embeddings(train_dataset)
    embeds["test"], labels["test"] = model.get_embeddings(test_dataset)
    embeds["train"] = embeds["train"].to("cpu").detach().numpy().copy()
    labels["train"] = labels["train"].to("cpu").detach().numpy().copy()
    embeds["test"] = embeds["test"].to("cpu").detach().numpy().copy()
    labels["test"] = labels["test"].to("cpu").detach().numpy().copy()

    for n in np.unique(labels["train"]):
        plt.scatter(
            embeds["train"][labels["train"] == n, 0],
            embeds["train"][labels["train"] == n, 1],
            s=1,
            label=f"cluster {n}",
        )
    plt.legend()
    plt.title("Embedding on training data")
    plt.show()

    for n in np.unique(labels["test"]):
        plt.scatter(
            embeds["test"][labels["test"] == n, 0],
            embeds["test"][labels["test"] == n, 1],
            s=1,
            label=f"cluster {n}",
        )
    plt.legend()
    plt.title("Embedding on test data")
    plt.show()


def main(cfg: DictConfig):
    """Perform inference."""
    device = get_device()
    init_manual_seed(0)  # fix seed

    embedding, _ = get_model(cfg, device)
    load_checkpoint(cfg, embedding)
    embedding.eval()
    train_dataset, test_dataset = get_dataset(cfg, 0)
    plot_result(train_dataset, test_dataset, embedding)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
