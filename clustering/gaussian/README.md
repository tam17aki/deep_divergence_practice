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
1. Edit `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths (`root_dir`) according to your environment.

2. Run `training.py`. It performs model training.

3. Run `inference.py`. It performs inference (prediction) using trained model.

You can change loss type and miner type by editing `config.yaml`.

4. Run `entropic_clustering.py`. It performs training and prediction of the method proposed by "Davis & Dhillon" [1].

You can run `plot_dataset.py` to plot the dataset. You can also run `plot_embedding.py` to plot embeddings with trained model.

## Results

* Rand Index (RI)

| Method (loss function)| Mean | Std dev. |
| --- | --- | --- |
| Davis & Dhillon | 0.649826 | 0.008840 |
| baseline</span> (constrative loss) | 0.619134 | 0.007905 |
| proposed</span> (constrative loss) | **0.987221** | 0.010623 |
| baseline</span> (triplet loss) | 0.657524 | 0.012132 |
| proposed</span> (triplet loss) | **0.986799** | 0.009837 |

* Adjusted Rand Index (ARI)

| Method (loss function)| Mean | Std dev. |
| --- | --- | --- |
| Davis & Dhillon | 0.241574 | 0.009874 |
| baseline (constrative loss) | 0.149356 | 0.018237 |
| proposed (constrative loss) | **0.971196** | 0.023947 |
| baseline (triplet loss) | 0.238092 | 0.029025 |
| proposed (triplet loss) | **0.970267** | 0.022193 |

I could reproduce almost the sample results. However, I consider that this experiment is **not a fair comparison** between the baseline and the proposed. Because the baseline clusters all 25,000 (= 500 * 50) points, whereas the proposed clusters only 500 points, putting the baseline at a huge disadvantage. We can obtain the above results with `training.py` and `inference.py`.

In order to make a fair comparison, **a mean vector was calculated for the baseline in units of the empirical distribution after embedding**, and the resulting 500 points were clustered. The results are shown in the following table.

| Metrics (loss function)| Mean | Std dev. |
| --- | --- | --- |
|RI (constrative loss) | 0.710910 | 0.020216 |
|RI (triplet loss) | 0.901598 | 0.022337 |
|ARI (constrative loss) | 0.360987  | 0.043097 |
|ARI (triplet loss) | 0.780103 | 0.049799 |

It can be seen that the baseline score has certainly improved, but it is still low compared to the proposed. We believe that these results have finally demonstrated the effectiveness of the proposed.
We can obtain the above results with `inference_fair_comparison.py`.

## Reference
[1] Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
