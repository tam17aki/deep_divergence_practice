# -*- coding: utf-8 -*-
"""Provides utilities.

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

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, homogeneity_score, roc_auc_score
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score, contingency_matrix,
                                     normalized_mutual_info_score, rand_score)
from sklearn.neighbors import KNeighborsClassifier


def init_manual_seed(random_seed: int):
    """Initialize manual seed."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def purity_score(y_true, y_pred):
    """Compute purity score."""
    contingency_mat = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)


def one_hot(labels, num_classes):
    """Convert numerical labels into one-hot vectors."""
    return np.squeeze(np.eye(num_classes)[labels.reshape(-1)])


def get_device():
    """Get a device specification."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    embeds["train"], labels["train"] = model.get_embeddings(train_dataset)
    embeds["test"], labels["test"] = model.get_embeddings(test_dataset)

    embeds["train"] = embeds["train"].to("cpu").detach().numpy().copy()
    embeds["test"] = embeds["test"].to("cpu").detach().numpy().copy()
    labels["train"] = labels["train"].to("cpu").detach().numpy().copy()
    labels["test"] = labels["test"].to("cpu").detach().numpy().copy()

    kmeans = KMeans(
        n_clusters=cfg.inference.n_clusters,
        n_init="auto",
        max_iter=cfg.inference.n_iter_kmeans,
        random_state=seed,
    )
    kmeans.fit(embeds["train"])
    pred_kmeans = kmeans.predict(embeds["test"])

    knn = KNeighborsClassifier(n_neighbors=cfg.inference.n_neighbors)
    knn.fit(embeds["train"], labels["train"])
    pred_knn = knn.predict(embeds["test"])

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


def init_stats():
    """Initialize stats dictionary."""
    stats = {
        "ri": [],  # Rand Index
        "ari": [],  # Adjusted Rand Index
        "ami": [],  # Adjusted Mutual Information
        "nmi": [],  # Normalized Mutual Information
        "purity": [],  # Purity Score
        "homo": [],  # Homogeneity Score
        "acc": [],  # Accuracy (classification)
        "auc_ovr": [],  # AUC (one-vs-rest)
        "auc_ovo": [],  # AUC (one-vs-one)
    }
    return stats


def append_stats(stats, new_stats):
    """Append new stats to dictionary."""
    stats["ri"].append(new_stats["ri"])
    stats["ari"].append(new_stats["ari"])
    stats["ami"].append(new_stats["ami"])
    stats["nmi"].append(new_stats["nmi"])
    stats["purity"].append(new_stats["purity"])
    stats["homo"].append(new_stats["homo"])
    stats["acc"].append(new_stats["acc"])
    stats["auc_ovr"].append(new_stats["auc_ovr"])
    stats["auc_ovo"].append(new_stats["auc_ovo"])


def print_stats(stats):
    """Print stats."""
    means = {
        "ri": None,  # Rand Index
        "ari": None,  # Adjusted Rand Index
        "ami": None,  # Adjusted Mutual Information
        "nmi": None,  # Normalized Mutual Information
        "purity": None,  # Purity Score
        "homo": None,  # Homogeneity Score
        "acc": None,  # Accuracy (classification)
        "auc_ovr": None,  # AUC (one-vs-rest)
        "auc_ovo": None,  # AUC (one-vs-one)
    }
    stds = {
        "ri": None,  # Rand Index
        "ari": None,  # Adjusted Rand Index
        "ami": None,  # Adjusted Mutual Information
        "nmi": None,  # Normalized Mutual Information
        "purity": None,  # Purity Score
        "homo": None,  # Homogeneity Score
        "acc": None,  # Accuracy (classification)
        "auc_ovr": None,  # AUC (one-vs-rest)
        "auc_ovo": None,  # AUC (one-vs-one)
    }

    means["ri"] = np.mean(np.array(stats["ri"]))
    means["ari"] = np.mean(np.array(stats["ari"]))
    means["ami"] = np.mean(np.array(stats["ami"]))
    means["nmi"] = np.mean(np.array(stats["nmi"]))
    means["purity"] = np.mean(np.array(stats["purity"]))
    means["homo"] = np.mean(np.array(stats["homo"]))
    means["acc"] = np.mean(np.array(stats["acc"]))
    means["auc_ovr"] = np.mean(np.array(stats["auc_ovr"]))
    means["auc_ovo"] = np.mean(np.array(stats["auc_ovo"]))

    stds["ri"] = np.std(np.array(stats["ri"]))
    stds["ari"] = np.std(np.array(stats["ari"]))
    stds["ami"] = np.std(np.array(stats["ami"]))
    stds["nmi"] = np.std(np.array(stats["nmi"]))
    stds["purity"] = np.std(np.array(stats["purity"]))
    stds["homo"] = np.std(np.array(stats["homo"]))
    stds["acc"] = np.std(np.array(stats["acc"]))
    stds["auc_ovr"] = np.std(np.array(stats["auc_ovr"]))
    stds["auc_ovo"] = np.std(np.array(stats["auc_ovo"]))

    print(
        "\nClustering:\n"
        f"Rand Index = {means['ri']:.06f} ± {stds['ri']:.06f}\n"
        f"Adjusted Rand Index = {means['ari']:.06f} ± {stds['ari']:.06f}\n"
        f"Adjusted Mutual Information = {means['ami']:.06f} ± {stds['ami']:.06f}\n"
        f"Normalized Mutual Information = {means['nmi']:.06f} ± {stds['nmi']:.06f}\n"
        f"Purity Score = {means['purity']:.06f} ± {stds['purity']:.06f}\n"
        f"Homogenity Score = {means['homo']:.06f} ± {stds['homo']:.06f}\n"
        "\nClassification:\n"
        f"Accuracy = {means['acc']:.06f} ± {stds['acc']:.06f}\n"
        f"ROC AUC (one-vs-rest) = {means['auc_ovr']:.06f} ± {stds['auc_ovr']:.06f}\n"
        f"ROC AUC (one-vs-one) = {means['auc_ovo']:.06f} ± {stds['auc_ovo']:.06f}\n"
    )


def load_checkpoint(cfg, model, seed=0):
    """Load checkpoint."""
    model_dir = os.path.join(cfg.directory.root_dir, cfg.directory.model_dir)
    if cfg.model.euc_dist is False:  # moment matching
        model_file = (
            cfg.training.loss_type + "_" + f"seed{seed}_" + cfg.training.model_file
        )
        model_file = os.path.join(model_dir, model_file)
    else:  # Euclidean distance
        model_file = (
            cfg.training.loss_type + "_" + f"seed{seed}_" + cfg.training.model_euc_file
        )
        model_file = os.path.join(model_dir, model_file)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)


def save_checkpoint(cfg, model, seed=0):
    """Save checkpoint."""
    model_dir = os.path.join(cfg.directory.root_dir, cfg.directory.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    if cfg.model.euc_dist is False:  # moment matching
        model_file = (
            cfg.training.loss_type + "_" + f"seed{seed}_" + cfg.training.model_file
        )
        model_file = os.path.join(model_dir, model_file)
    else:  # Euclidean distance
        model_file = (
            cfg.training.loss_type + "_" + f"seed{seed}_" + cfg.training.model_euc_file
        )
        model_file = os.path.join(model_dir, model_file)
    torch.save(model.state_dict(), model_file)
