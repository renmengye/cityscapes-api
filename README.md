# cityscapes-api
API for Cityscapes Dataset

## Usage
```python
from cityscapes_api import get_dataset, get_iterator

train_iter = get_iterator(get_dataset("cityscapes", "train"), batch_size=4)
val_iter = get_iterator(
    get_dataset("cityscapes", "valid"), batch_size=4, cycle=False)

# Infinite training loop.
for batch in train_iter:
  image = batch["input"]
  label = batch["label_sem_seg"]
  # Train the network with image and label.

# Evaluate the network for one epoch.
for batch in val_iter:
  image = batch["input"]
  label = batch["label_sem_seg"]
  # Run validation on the current mini-batch.
```
