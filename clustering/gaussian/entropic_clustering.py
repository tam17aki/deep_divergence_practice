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
from numba import jit
from numpy import linalg as LA
from omegaconf import OmegaConf
from progressbar import progressbar as prg
from sklearn.metrics import accuracy_score, homogeneity_score, roc_auc_score
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     normalized_mutual_info_score, rand_score)

from dataset import make_circles_triple
from util import append_stats, init_stats, one_hot, print_stats, purity_score


@jit(nopython=True)
def comp_burg_div(mat_x, mat_y):
    """Compute Burg matrix divergence."""
    dim = mat_x.shape[0]
    mat_y_inv = LA.inv(mat_y)
    mat = mat_x @ mat_y_inv
    burg_div = np.trace(mat) - np.log(LA.det(mat)) - dim
    return burg_div


@jit(nopython=True)
def comp_maha_dist(vec_x, vec_y, cov):
    """Compute Mahalanobis distance."""
    delta = vec_x - vec_y
    dist = np.dot(np.dot(delta, np.atleast_2d(LA.inv(cov))), delta)
    return np.sqrt(dist)


class EntropicClustering:
    """Differential entropic clustering.

    Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
    """

    def __init__(self, cfg, seed):
        """Initialize class."""
        self.n_clusters = cfg.training.n_clusters
        self.n_samples = cfg.training.n_train
        self.n_epoch = cfg.training.n_epoch
        rng = np.random.default_rng(seed)
        self.assign = rng.choice(self.n_clusters, self.n_samples)  # cluster assignments
        self.stats = {"mean": None, "cov": None}
        self.cluster_means = [None] * self.n_clusters
        self.cluster_covs = [None] * self.n_clusters

    def _estep(self):
        """Assign each Gaussian to the closest cluster representative Gaussian."""
        for i in range(self.n_samples):
            assign = []
            for j in range(self.n_clusters):
                burg_div = comp_burg_div(self.stats["cov"][i], self.cluster_covs[j])
                maha_dist = comp_maha_dist(
                    self.stats["mean"][i], self.cluster_means[j], self.cluster_covs[j]
                )
                assign.append(burg_div + maha_dist)
            self.assign[i] = np.argmin(np.array(assign))

    def _mstep(self):
        """Update cluster means and cluster covariances."""
        for j in range(self.n_clusters):
            idx = np.argwhere(self.assign == j).squeeze()
            n_count = len(np.where(self.assign == j)[0])
            if n_count == 0:
                continue
            if n_count == 1:
                idx = [idx]

            cluster_means = np.zeros_like(self.stats["mean"][0])
            for i in idx:
                cluster_means += self.stats["mean"][i]
            self.cluster_means[j] = cluster_means / n_count

            cluster_covs = np.zeros_like(self.stats["cov"][0])
            for i in idx:
                diff = self.stats["mean"][i] - self.cluster_means[j]
                cluster_covs += self.stats["cov"][i] + diff * diff.T
            self.cluster_covs[j] = cluster_covs / n_count

    def fit(self, means, covs):
        """Fit module.

        Args:
           means: list of mean vectors.
           covs: list of covariance matrices.
        """
        self.stats = {"mean": means, "cov": covs}
        for _ in range(self.n_epoch):
            self._mstep()  # update cluster means/covariances
            self._estep()  # update assignment

    def predict(self, means, covs):
        """Predict assignment.

        Args:
            means: list of mean vectors.
            covs: list of covariance matrices.
        """
        n_samples = len(means)
        assignment = [0] * n_samples
        for i in range(n_samples):
            assign = []
            for j in range(self.n_clusters):
                burg_div = comp_burg_div(covs[i], self.cluster_covs[j])
                maha_dist = comp_maha_dist(
                    means[i], self.cluster_means[j], self.cluster_covs[j]
                )
                assign.append(burg_div + maha_dist)
            assignment[i] = np.argmin(np.array(assign))
        return np.array(assignment)


def get_dataset(cfg, seed):
    """Instantiate dataset."""
    rng = np.random.default_rng(seed=seed)
    train_mean, train_label = make_circles_triple(
        cfg.training.n_train,
        noise=cfg.dataset.circles_noise,
        random_state=seed,
        factors=cfg.dataset.factors,
    )
    test_mean, test_label = make_circles_triple(
        cfg.training.n_test,
        noise=cfg.dataset.circles_noise,
        random_state=seed,
        factors=cfg.dataset.factors,
    )
    point_list_train = [
        train_mean[i]
        + np.sqrt(cfg.dataset.gauss_cov)
        * rng.standard_normal(size=(cfg.training.n_points, 2))
        for i in range(cfg.training.n_train)
    ]  # e.g., 500 * 50 = 25,000 points in 2-D
    point_list_test = [
        test_mean[i]
        + np.sqrt(cfg.dataset.gauss_cov)
        * rng.standard_normal(size=(cfg.training.n_points, 2))
        for i in range(cfg.training.n_test)
    ]  # e.g., 200 * 50 = 10,000 points in 2-D
    train_stats = {"mean": None, "cov": None}
    train_stats["mean"] = [np.mean(points, axis=0) for points in point_list_train]
    train_stats["cov"] = [np.cov(points, rowvar=False) for points in point_list_train]
    test_stats = {"mean": None, "cov": None}
    test_stats["mean"] = [np.mean(points, axis=0) for points in point_list_test]
    test_stats["cov"] = [np.cov(points, rowvar=False) for points in point_list_test]
    return train_stats, train_label, test_stats, test_label


def calc_accuracy(cfg, label, stats, module):
    """Compute various accuracy metrics.

    Args:
        label: reference labels.
        stats (dict): means and covariances for each observation.
    """
    pred = module.predict(stats["mean"], stats["cov"])
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


def main(cfg):
    """Perform training and calculate metric accuracies."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    # perform training loops changing random seed for dataset
    metrics = {"train": init_stats(), "test": init_stats()}
    for seed in prg(range(cfg.training.n_trial)):
        train_stats, train_label, test_stats, test_label = get_dataset(cfg, seed)
        module = EntropicClustering(cfg, seed)
        module.fit(train_stats["mean"], train_stats["cov"])
        append_stats(
            metrics["train"], calc_accuracy(cfg, train_label, train_stats, module)
        )
        append_stats(
            metrics["test"], calc_accuracy(cfg, test_label, test_stats, module)
        )
    print("\nResult (Training)", end="")
    print_stats(metrics["train"])
    print("\nResult (Test)", end="")
    print_stats(metrics["train"])


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
