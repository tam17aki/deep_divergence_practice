"""Differential Entropic Clustering of Multivariate Gaussians.

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
from hydra import compose, initialize
from numpy import linalg as LA
from omegaconf import OmegaConf
from progressbar import progressbar as prg
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, homogeneity_score, roc_auc_score
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     normalized_mutual_info_score, rand_score)

from dataset import make_circles_triple
from util import append_stats, init_stats, one_hot, print_stats, purity_score


def comp_burg_div(mat_x, mat_y, dim):
    """Compute Burg matrix divergence."""
    mat_y_inv = LA.inv(mat_y)
    mat = mat_x @ mat_y_inv
    burg_div = np.trace(mat) - np.log(LA.det(mat)) - dim
    return burg_div


def comp_maha_dist(vec_a, vec_b, cov):
    """Compute Mahalanobis distance."""
    return distance.mahalanobis(vec_a, vec_b, LA.inv(cov))


class EntropicClustering:
    """Differential entropic clustering.

    Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
    """

    def __init__(self, dataset, n_clusters):
        """Initialize class."""
        # a dictionary of stats from empirical distribution
        self.stats = dataset["stats"]
        self.labels = dataset["labels"]
        self.n_clusters = n_clusters

        self.n_samples = len(self.stats["mean"])
        rng = np.random.default_rng(0)
        self.gamma = rng.choice(self.n_clusters, self.n_samples)  # cluster assignments

        self.cluster_means = [None] * self.n_clusters
        self.cluster_covs = [None] * self.n_clusters

    def estep(self):
        """Assign each Gaussian to the closest cluster representative Gaussian."""
        dim = self.stats["cov"][0].shape[0]
        for i in range(self.n_samples):
            assign = []
            for j in range(self.n_clusters):
                burg_div = comp_burg_div(
                    self.stats["cov"][i], self.cluster_covs[j], dim
                )
                maha_dist = comp_maha_dist(
                    self.stats["mean"][i], self.cluster_means[j], self.cluster_covs[j]
                )
                assign.append(burg_div + maha_dist)
            self.gamma[i] = np.argmin(np.array(assign))

    def mstep(self):
        """Update cluster means and cluster covariances."""
        for j in range(self.n_clusters):
            idx = np.argwhere(self.gamma == j).squeeze()
            n_count = len(idx)
            if n_count == 0:
                continue

            cluster_means = np.zeros_like(self.stats["mean"][0])
            for i in idx:
                cluster_means += self.stats["mean"][i]
            self.cluster_means[j] = cluster_means / n_count

            cluster_covs = np.zeros_like(self.stats["cov"][0])
            for i in idx:
                diff = self.stats["mean"][i] - self.cluster_means[j]
                cluster_covs += self.stats["cov"][i] + diff * diff.T
            self.cluster_covs[j] = cluster_covs / n_count

    def inference(self, stats):
        """Perform inference of assignment.

        Args:
           stats: dictironary of mean and cov.
        """
        dim = stats["cov"][0].shape[0]
        n_samples = len(stats["mean"])
        gamma = [0] * n_samples
        for i in range(n_samples):
            assign = []
            for k in range(self.n_clusters):
                burg_div = comp_burg_div(stats["cov"][i], self.cluster_covs[k], dim)
                maha_dist = comp_maha_dist(
                    stats["mean"][i], self.cluster_means[k], self.cluster_covs[k]
                )
                assign.append(burg_div + maha_dist)
            gamma[i] = np.argmin(np.array(assign))
        return np.array(gamma)


def get_dataset(cfg, seed):
    """Instantiate dataset."""
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

    stats_train = {"mean": None, "cov": None}
    stats_train["mean"] = [np.mean(point, axis=0) for point in point_list_train]
    stats_train["cov"] = [np.cov(point, rowvar=False) for point in point_list_train]
    stats_test = {"mean": None, "cov": None}
    stats_test["mean"] = [np.mean(point, axis=0) for point in point_list_test]
    stats_test["cov"] = [np.cov(point, rowvar=False) for point in point_list_test]
    train_dataset = {"stats": stats_train, "labels": train_label}
    test_dataset = {"stats": stats_test, "labels": test_label}
    return train_dataset, test_dataset


def calc_accuracy(cfg, dataset, module):
    """Compute various accuracy metrics.

    Args:
        cfg: configuration.
        dataset (dict): dictionary of stats and labels.
        module (EntropicClustering): clustering module.
    """
    label = dataset["labels"]
    pred = module.inference(dataset["stats"])
    return {
        "ri": rand_score(label, pred),
        "ari": adjusted_rand_score(label, pred),
        "ami": adjusted_mutual_info_score(label, pred),
        "nmi": normalized_mutual_info_score(label, pred),
        "purity": purity_score(label, pred),
        "homo": homogeneity_score(label, pred),
        "acc": accuracy_score(label, pred),
        "auc_ovr": roc_auc_score(
            label, one_hot(pred, cfg.inference.n_clusters), multi_class="ovr"
        ),
        "auc_ovo": roc_auc_score(
            label, one_hot(pred, cfg.inference.n_clusters), multi_class="ovo"
        ),
    }


def training_loop(cfg, module):
    """Perform training loop.

    Args:
        cfg: configuration.
        module (EntropicClustering): clustering module.
    """
    for _ in range(cfg.training.n_epoch):
        # perform differential entropic clustering
        module.mstep()  # update cluster means/covariances
        module.estep()  # update assignment


def main(cfg):
    """Perform training and calculate metric accuracies."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    # perform training loops changing random seed for dataset
    all_stats = init_stats()
    for seed in prg(range(cfg.training.n_trial)):
        train_dataset, test_dataset = get_dataset(cfg, seed)
        module = EntropicClustering(train_dataset, cfg.training.n_clusters)
        training_loop(cfg, module)
        append_stats(all_stats, calc_accuracy(cfg, test_dataset, module))
    print_stats(all_stats)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
