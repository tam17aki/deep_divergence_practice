## Dependencies
We tested the implemention on Ubuntu 22.04. The verion of Python was `3.10.12`. The following modules are required:

- torch
- hydra-core
- progressbar2
- numpy
- sklearn
- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning#installation) 
- faiss-gpu https://pypi.org/project/faiss-gpu/


## Recipes
1. Modify `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths (`root_dir`) according to your environment.

2. Run `training.py`. It performs model training.

3. Run `inference.py`. It performs inference (prediction) using trained model.

You can change loss type and miner type by editing `config.yaml`.

4. Run `entropic_clustering.py`. It performs training and prediction of the method proposed by "Davis & Dhillon" [1].

You can run `plot_dataset.py` to plot the dataset. You can also run `plot_embedding.py` to plot embeddings with trained model.

## Reference
[1] Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
