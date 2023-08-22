# -*- coding: utf-8 -*-
"""Plot point cloud used in the experiment.

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

from dataset import make_circles_triple


def plot_cloud(cfg):
    """hoge."""
    mean, labels = make_circles_triple(
        cfg.training.n_train,
        noise=cfg.dataset.circles_noise,
        random_state=42,
        factors=cfg.dataset.factors,
    )
    fig = plt.figure(figsize=(25, 4))
    axes = [None, None, None, None, None]
    axes[0] = fig.add_subplot(1, 5, 1)
    for lab in [2, 1, 0]:
        idx = np.where(labels == lab)[0]
        axes[0].scatter(mean[idx, 0], mean[idx, 1], s=1)

    rng = np.random.default_rng(seed=0)
    point_list = [
        mean[i]
        + np.sqrt(cfg.dataset.gauss_cov)
        * rng.standard_normal(size=(cfg.training.n_points, 2))
        for i in range(cfg.training.n_train)
    ]
    points = np.concatenate(point_list, axis=0)
    labels = np.repeat(labels, cfg.training.n_points)
    color_list = ["C2", "C1", "C0"]
    for i in range(3):
        axes[i + 1] = fig.add_subplot(1, 5, i + 2)
        idx = np.where(labels == i)[0]
        axes[i + 1].scatter(points[idx, 0], points[idx, 1], s=1, color=color_list[i])
        axes[i + 1].set_xlim(-2.2, 2.2)
        axes[i + 1].set_ylim(-2.2, 2.2)
    axes[4] = fig.add_subplot(1, 5, 5)
    for lab in [2, 1, 0]:
        idx = np.where(labels == lab)[0]
        axes[4].scatter(points[idx, 0], points[idx, 1], s=1)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    plot_cloud(config)
