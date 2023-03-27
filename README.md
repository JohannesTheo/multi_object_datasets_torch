# multi_object_datasets_torch

This repository is WIP

### install dependencies

 ```bash
 # mandatory: install these manually
 pip install tensorflow-cpu
 pip install torch torchvision

 # optional: the following deps are installed automatically but we show them here for transparency
 pip install h5py, tqdm, numpy, gsutil 
 pip install git+https://github.com/deepmind/multi_object_datasets.git@master
 ```

### install repository

```bash
# for usage:
pip install git+https://github.com/JohannesTheo/multi_object_datasets_torch.git@main

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

### TODOs

Before release:

- TODO: Write proper README
- TODO: Add licence
- TODO: Add support for torchvision transforms
- TODO: Add example for Clevr Crop as used in IODINE, Genesis

After release:

- TODO: Set good default values for split size (currently same as object-centric lib)
- TODO: Port segmentation_metrics.py
- TODO: test with different (older) versions of python, tf and torch, (h5, gsutil?)
