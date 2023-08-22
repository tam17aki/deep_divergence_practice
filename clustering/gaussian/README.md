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

3. Run `inference.py`. It performs inference using trained model (i.e., generate audios from onomatopoeia).
