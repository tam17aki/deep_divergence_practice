# Clustering Multivariate Gaussian Distributions

## Recipe
1. Edit `config.yaml` in accordance with your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths (`root_dir`).

2. Run `training.py`. It performs model training.

3. Run `inference.py`. It performs inference (prediction) using trained model.

You can change loss type and miner type by editing `config.yaml`.

4. Run `entropic_clustering.py`. It performs training and prediction of the method proposed by "Davis & Dhillon" [1].



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

## Bonus

### Visualization of dataset

You can run `plot_dataset.py` to plot the dataset. 

![Figure 1](https://github.com/tam17aki/deep_divergence_practice/blob/main/clustering/gaussian/img/dataset.png?raw=true)

We can generate **500** Gaussian mean vectors around the circumference of three concentric circles with different radii (for training; another 200 for testing). The radii of the concentric circles are **0.2, 0.6, 1.0**, respectively, and the three concentric circles correspond to the three clusters, respectively. One mean vector is obtained by randomly sampling points on the circle and superimposing two-dimensional Gaussian noise. The size of the noise at this time was not specified in the paper, but in this experiment, it was set to **standard deviation 0.05** in each dimension. The above corresponds to the leftmost part of the figure. Coloring makes it easier to see the corresponding clusters.

Then randomly sample a number of points from each Gaussian distribution; **A diagonal covariance matrix is assumed and each diagonal element is set to 0.1**. Multiple 2D Gaussian noises scaled by $\sqrt{0.1}$ are generated independently and added to each mean vector. Although the paper did not specify how many points to sample, Figure 1 in the paper shows a scatter plot after sampling 50 points from each Gaussian distribution, I also sampled **50** points per Gaussian distribution. The middle three scatterplots in the figure show point clouds randomly sampled from the Gaussian distribution of each cluster. On the far right is the superposition of all three scatterplots. Note that it has a two-step structure, sampling the mean vector and sampling from the corresponding Gaussian distribution.

Finally, we have a training point cloud of $500 \times 50 = 25,000$ points and a test point cloud of $200 \times 50 = 10,000$ points, each of which belongs to one of three clusters (= is labeled). As can be seen from the figure, the point clouds of each cluster overlap each other significantly due to the large variance.

### Visualization of embeddings

You can run `plot_embedding.py` to plot embeddings with trained model.

The following figure shows an embedding visualization for the baseline. It is a visualization on the training data, and the embedding network trained with the triplet loss function is used.

![Figure 2](https://github.com/tam17aki/deep_divergence_practice/blob/main/clustering/gaussian/img/triplet_embed_train_euc.png?raw=true)

Next, the following shows an embedding visualization for the proposed. It is also a visualization on the training data, and the network was also trained with the same loss funciton.

![Figure 3](https://github.com/tam17aki/deep_divergence_practice/blob/main/clustering/gaussian/img/triplet_embed_train.png?raw=true)

## Reference
[1] Jason V. Davis and Inderjit Dhillon, "Differential entropic
    clustering of multivariate Gaussians," In Advances in Neural
    Information Processing Systems (NIPS), 2006.
