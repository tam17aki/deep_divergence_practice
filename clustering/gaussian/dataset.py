# -*- coding: utf-8 -*-
"""Dataset.

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
from sklearn.utils import shuffle as util_shuffle
from torch.utils.data import TensorDataset


def make_circles_triple(
    n_samples=100, shuffle=True, noise=None, random_state=None, factors=(0.2, 0.6, 1.0)
):
    """Make a large circle containing 'two' smaller circles in 2d.

    A simple toy dataset to visualize clustering and classification algorithms.

    Parameters
    ----------
    n_samples : int, default=100
        It is the total number of points generated.

    shuffle : bool, default=True
        Whether to shuffle the samples.

    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.

    random_state : int, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.

    factors : tuple, default=(0.1, 0.6, 1.1)
        Tuple of scale factors of inner, middle, and outer circle in the range `[0, 1)`.

    Returns
    -------
    data : ndarray of shape (n_samples, 2)
        The generated samples.

    label : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    if len(factors) != 3:
        raise ValueError("factors must be a tuple of three float elements.")
    generator = np.random.default_rng(random_state)
    rand = generator.choice(3, n_samples)  # a sequence of (0, 1, 2)
    data = []
    label = []
    for i, factor in enumerate(factors):
        n_samples = len(np.where(rand == i)[0])
        linspace = generator.uniform(0, 2 * np.pi, n_samples)
        data.append(
            np.concatenate(
                (
                    np.expand_dims(factor * np.cos(linspace), axis=1),
                    np.expand_dims(factor * np.sin(linspace), axis=1),
                ),
                axis=1,
            )
        )
        label.append(i * np.ones(n_samples, dtype=np.intp))
    data = np.concatenate(data)
    label = np.concatenate(label)
    if shuffle:
        data, label = util_shuffle(data, label, random_state=random_state)
    if noise is not None:
        data += generator.normal(scale=noise, size=data.shape)
    return data, label


class DistributionDataset(torch.utils.data.Dataset):
    """Dataset for empirical distribution."""

    def __init__(self, points, labels):
        """Initialize class."""
        self.points = points  # a list of points from empirical distribution
        self.labels = labels

    @property
    def get_points(self):
        """Return points of empirical distribution."""
        return np.concatenate(self.points)

    @property
    def get_labels(self):
        """Return labels."""
        return np.copy(self.labels)

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get a pair of input and target.

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        points = np.array(self.points[idx])
        label = self.labels[idx]
        return (points, label)


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
    if cfg.model.euc_dist is True:
        train_data = torch.tensor(np.concatenate(point_list_train, axis=0))
        train_label = torch.tensor(np.repeat(train_label, cfg.training.n_points))
        test_data = torch.tensor(np.concatenate(point_list_test, axis=0))
        test_label = torch.tensor(np.repeat(test_label, cfg.training.n_points))
        train_dataset = TensorDataset(train_data, train_label)
        test_dataset = TensorDataset(test_data, test_label)
    else:  # moment-matching
        train_dataset = DistributionDataset(point_list_train, train_label)
        test_dataset = DistributionDataset(point_list_test, test_label)
    return train_dataset, test_dataset
