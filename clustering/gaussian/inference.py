# -*- coding: utf-8 -*-
"""Inference script.

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
import warnings

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from progressbar import progressbar as prg

from dataset import get_dataset
from model import get_model
from util import (append_stats, calc_accuracy, get_device, init_manual_seed,
                  init_stats, load_checkpoint, print_stats)

warnings.simplefilter("ignore", UserWarning)


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
