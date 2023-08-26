# deep_divergence_practice
This repostitory provides implementations of "Deep Divergence Learning" [1] to reproduce the experimental results.

## Licence
MIT licence.

Copyright (C) 2023 Akira Tamamori

## Dependencies
We tested the implemention on Ubuntu 22.04. The verion of Python was `3.10.12`. The following modules are required:

- PyTorch (torch)
- Hydra (hydra-core)
- progressbar2
- Numba (numba)
- Numpy (numpy)
- scikit-learn (sklearn)
- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning#installation) 
- faiss-gpu https://pypi.org/project/faiss-gpu/

## Directory
- `clustering` ... Corresponding to the Section 5.1.

## References
[1] H. K. Cilingir, R. Manzelli, and B. Kulis, “Deep divergence learning,” in ICML, 2020.　http://proceedings.mlr.press/v119/cilingir20a.html
