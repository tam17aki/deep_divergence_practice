# -*- coding: utf-8 -*-
"""Inference script for fair comparison.

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
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from progressbar import progressbar as prg
from sklearn.metrics import accuracy_score, homogeneity_score, roc_auc_score
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     normalized_mutual_info_score, rand_score)

from dataset import DistributionDataset, make_circles_triple
from model import get_model
from util import (KMeans, KNeighborsClassifier, append_stats, get_device,
                  init_manual_seed, init_stats, load_checkpoint, one_hot,
                  print_stats, purity_score)


def get_dataset(cfg, seed):
    """Instantiate dataset.

    Args:
        seed (int): random seed for dataset.

    Returns:
        - train_dataset (DistributionDataset): dataset for training
          or train_dataset (TensorDataset): dataset for training (Euclidean distance)
        - test_dataset (DistributionDataset): dataset for test
          or test_dataset (TensorDataset): dataset for test  (Euclidean distance)
    """
    rng = np.random.default_rng(seed=seed)
    mean, train_label = make_circles_triple(
        cfg.training.n_train,
        noise=cfg.dataset.circles_noise,
        random_state=seed,
        factors=cfg.dataset.factors,
    )
    point_list_train = [
        mean[i]
        + np.sqrt(cfg.dataset.gauss_cov)
        * rng.standard_normal(size=(cfg.training.n_points, 2))
        for i in range(cfg.training.n_train)
    ]  # e.g., 500 * 50 = 25,000 points in 2-D
    mean, test_label = make_circles_triple(
        cfg.training.n_test,
        noise=cfg.dataset.circles_noise,
        random_state=seed,
        factors=cfg.dataset.factors,
    )
    point_list_test = [
        mean[i]
        + np.sqrt(cfg.dataset.gauss_cov)
        * rng.standard_normal(size=(cfg.training.n_points, 2))
        for i in range(cfg.training.n_test)
    ]  # e.g., 200 * 50 = 10,000 points in 2-D
    train_dataset = DistributionDataset(point_list_train, train_label)
    test_dataset = DistributionDataset(point_list_test, test_label)
    return train_dataset, test_dataset


def get_embeddings(cfg, model, dataset):
    """Return embeddings.

    This method is intended to run in inference mode.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.inference.n_batch,
        shuffle=False,
        drop_last=False,
    )
    begin = 0
    end = 0
    for i, batch in enumerate(dataloader):
        data, label = batch
        data, label = data.float().to(device), label.long().to(device)
        embed = model.forward(data)
        embed = embed.mean(dim=1)
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


def calc_accuracy(cfg, train_dataset, test_dataset, model, seed=0):
    """Compute various accuracy metrics.

    Args:
        cfg: configuration.
        train_dataset: training (validation) dataset.
        test_dataset: test dataset.
        model: embedding function via neural net.
        seed (int) : random seed.
    """
    embeds = {"train": None, "test": None}
    labels = {"train": None, "test": None}

    model.eval()
    embeds["train"], labels["train"] = get_embeddings(cfg, model, train_dataset)
    embeds["test"], labels["test"] = get_embeddings(cfg, model, test_dataset)

    kmeans = KMeans(
        n_clusters=cfg.inference.n_clusters,
        niter=cfg.inference.n_iter_kmeans,
        seed=seed,
        gpu=True,
    )
    kmeans.fit(embeds["train"])
    pred_kmeans = kmeans.predict(embeds["test"])
    pred_kmeans = pred_kmeans.to("cpu").detach().numpy().copy()

    knn = KNeighborsClassifier(k=cfg.inference.top_k)
    knn.fit(embeds["train"], labels["train"])
    pred_knn = knn.predict(embeds["test"])
    pred_knn = pred_knn.to("cpu").detach().numpy().copy()

    labels["test"] = labels["test"].to("cpu").detach().numpy().copy()
    return {
        "ri": rand_score(labels["test"], pred_kmeans),
        "ari": adjusted_rand_score(labels["test"], pred_kmeans),
        "ami": adjusted_mutual_info_score(labels["test"], pred_kmeans),
        "nmi": normalized_mutual_info_score(labels["test"], pred_kmeans),
        "purity": purity_score(labels["test"], pred_kmeans),
        "homo": homogeneity_score(labels["test"], pred_kmeans),
        "acc": accuracy_score(labels["test"], pred_knn),
        "auc_ovr": roc_auc_score(
            labels["test"],
            one_hot(pred_knn, cfg.inference.n_clusters),
            multi_class="ovr",
        ),
        "auc_ovo": roc_auc_score(
            labels["test"],
            one_hot(pred_knn, cfg.inference.n_clusters),
            multi_class="ovo",
        ),
    }


def main(cfg: DictConfig):
    """Perform inference."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration
    device = get_device()
    init_manual_seed(0)  # fix seed

    all_stats = init_stats()
    for seed in prg(range(cfg.inference.n_trial)):
        train_dataset, test_dataset = get_dataset(cfg, seed)
        embedding, _ = get_model(cfg, device)
        load_checkpoint(cfg, embedding, seed)
        append_stats(
            all_stats,
            calc_accuracy(cfg, train_dataset, test_dataset, embedding, seed),
        )

    print_stats(all_stats)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
