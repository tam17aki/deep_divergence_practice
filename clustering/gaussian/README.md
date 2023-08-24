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
| proposed</span> (constrative loss) | **0.988035** | 0.011927 |
| baseline</span> (triplet loss) | 0.657529 | 0.012002 |
| proposed</span> (triplet loss) | **0.991387** | 0.009460 |

* Adjusted Rand Index (ARI)

| Method (loss function)| Mean | Std dev. |
| --- | --- | --- |
| Davis & Dhillon | 0.241574 | 0.009874 |
| baseline (constrative loss) | 0.238745 | 0.030552 |
| proposed (constrative loss) | **0.973040** | 0.026887 |
| baseline (triplet loss) | 0.238126 | 0.028162 |
| proposed (triplet loss) | **0.980598** | 0.021340 |

I could reproduce almost the sample results. However, I consider that this experiment is **not a fair comparison** between the baseline and the proposed method. Because the baseline clusters all 25,000 (= 500 * 50) points, whereas the proposed method clusters only 500 points, putting the baseline at a huge disadvantage. We can obtain the above results with `training.py` and `inference.py`.

In order to make a fair comparison, **a mean vector was calculated for the baseline in units of the empirical distribution after embedding**, and the resulting 500 points were clustered. The results are shown in the following table.

| Metrics (loss function)| Mean | Std dev. |
| --- | --- | --- |
|RI (constrative loss) | 0.972352 | 0.017127 |
|RI (triplet loss) | 0.986995 | 0.005775 |
|ARI (constrative loss) | 0.937757 | 0.038647 |
|ARI (triplet loss) | 0.970741 | 0.013020 |

It can be seen that the scores has been improved significantly and **achieve almost equal level compared to the proposed method**, but the "average" was still slightly worse. I believe that these results have finally demonstrated the effectiveness of the proposed method.
We can obtain the above results with `inference_fair_comparison.py`. For visualization of embeddings, you can use `plot_embedding_fair_comparison.py`.

## Reference
[1] Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
