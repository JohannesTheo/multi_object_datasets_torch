# multi_object_datasets_torch

- This repository is WIP 
- TODO: Write proper README

# install and dependencies

```bash
pip install tensorflow-cpu && \
pip install torch torchvision && \
git clone https://github.com/JohannesTheo/multi_object_datasets_torch.git && \
pip install ./multi_object_datasets_torch/
```

- TODO: test with different dependency versions, e.g. tf and torch but also gsutil

# code

- TODO: Add support for torchvision transforms 
- TODO: Compare default splits from research papers and existing repos to set good default values

```python
from multi_object_datasets_torch import CaterWithMasks, ClevrWithMasks, MultiDSprites, ObjectsRoom, Tetrominoes

t_train = Tetrominoes("~/datasets")
print(t_train)
```
