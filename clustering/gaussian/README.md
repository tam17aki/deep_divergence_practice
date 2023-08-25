# Clustering Multivariate Gaussian Distributions

## Recipes
1. Edit `config.yaml` in accordance with your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths (`root_dir`).

2. Run `training.py`. It performs model training.

3. Run `inference.py`. It performs inference (prediction) using trained model.

You can change loss type and miner type by editing `config.yaml`.

4. Run `entropic_clustering.py`. It performs training and prediction of the method proposed by "Davis & Dhillon" [1].

You can run `plot_dataset.py` to plot the dataset. You can also run `plot_embedding.py` to plot embeddings with trained model.

## Results

The experimental results are shown in the following; averaged over 10 runs while changing random seed value.

* Rand Index (RI)

| Method (loss function)| Mean | Std dev. |
| --- | --- | --- |
| Davis & Dhillon | 0.649826 | 0.008840 |
| baseline</span> (constrative loss) | 0.650168 | 0.013206 |
| proposed</span> (constrative loss) | **0.990864** | 0.007971 |
| baseline</span> (triplet loss) | 0.657529 | 0.012002 |
| proposed</span> (triplet loss) | **0.992834** | 0.006162 |

* Adjusted Rand Index (ARI)

| Method (loss function)| Mean | Std dev. |
| --- | --- | --- |
| Davis & Dhillon | 0.241574 | 0.009874 |
| baseline (constrative loss) | 0.238745 | 0.030552 |
| proposed (constrative loss) | **0.979434** | 0.017977 |
| baseline (triplet loss) | 0.238126 | 0.028162 |
| proposed (triplet loss) | **0.983881** | 0.013887 |

I could reproduce almost the same results.

## Reference
[1] Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
