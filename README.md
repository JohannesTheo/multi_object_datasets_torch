# multi_object_datasets_torch

This repository is WIP

- TODO: Write proper README

### install dependencies

- TODO: test with different (older) versions, in particular python, tf and torch

 ```bash
 # mandatory: install these manually
 pip install tensorflow-cpu
 pip install torch torchvision

 # optional: the following deps are installed automatically but we show them here for transparency
 pip install h5py, tqdm, gsutil 
 pip install git+https://github.com/deepmind/multi_object_datasets.git@main
 ```

### install repository

```bash
# for usage:
pip install git+https://github.com/JohannesTheo/multi_object_datasets_torch.git@master
 
# for development:
git clone https://github.com/JohannesTheo/multi_object_datasets_torch.git && \
pip install -e ./multi_object_datasets_torch/
 ```

### usage

```python
from multi_object_datasets_torch import CaterWithMasks, ClevrWithMasks, MultiDSprites, ObjectsRoom, Tetrominoes

t_train = Tetrominoes("~/datasets")
print(t_train)
```

### code TODOs

- TODO: Add support for torchvision transforms 
- TODO: Compare default splits from research papers and existing repos to set good default values
- TODO: Port segmentation_metrics.py to torch
