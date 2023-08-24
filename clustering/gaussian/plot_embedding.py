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
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig

from dataset import get_dataset
from model import get_model
from util import get_device, init_manual_seed, load_checkpoint


def plot_result(cfg, train_dataset, test_dataset, model):
    """Plot embeddings."""
    embeds = {"train": None, "test": None}
    labels = {"train": None, "test": None}

    model.eval()
    embeds["train"], labels["train"] = model.get_embeddings(train_dataset)
    embeds["test"], labels["test"] = model.get_embeddings(test_dataset)

    embeds_train = embeds["train"].to("cpu").detach().numpy().copy()
    labels_train = labels["train"].to("cpu").detach().numpy().copy()
    embeds_test = embeds["test"].to("cpu").detach().numpy().copy()
    labels_test = labels["test"].to("cpu").detach().numpy().copy()

    for cluster_id in np.unique(labels_train):
        plt.scatter(
            embeds_train[labels_train == cluster_id, 0],
            embeds_train[labels_train == cluster_id, 1],
            s=1,
            label=f"cluster {cluster_id}",
        )
    plt.legend()
    plt.title(f"Embedding on train data ({cfg.training.loss_type} loss)")
    if cfg.model.euc_dist is False:
        plt.savefig(cfg.training.loss_type + "_" + cfg.inference.embed_fig_file)
    else:
        plt.savefig(cfg.training.loss_type + "_" + cfg.inference.embed_fig_euc_file)
    plt.show()

    for cluster_id in np.unique(labels_test):
        plt.scatter(
            embeds_test[labels_test == cluster_id, 0],
            embeds_test[labels_test == cluster_id, 1],
            s=1,
            label=f"cluster {cluster_id}",
        )
    plt.legend()
    plt.title(f"Embedding on test data ({cfg.training.loss_type} loss)")
    if cfg.model.euc_dist is False:
        plt.savefig(cfg.training.loss_type + "_" + cfg.inference.embed_test_fig_file)
    else:
        plt.savefig(
            cfg.training.loss_type + "_" + cfg.inference.embed_test_fig_euc_file
        )
    plt.show()


def main(cfg: DictConfig):
    """Perform inference."""
    device = get_device()
    init_manual_seed(0)  # fix seed

    embedding, _ = get_model(cfg, device)
    load_checkpoint(cfg, embedding)
    embedding.eval()
    train_dataset, test_dataset = get_dataset(cfg, 0)
    plot_result(cfg, train_dataset, test_dataset, embedding)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
