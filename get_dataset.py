from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from .batch_iter import BatchIterator
from .concurrent_batch_iter import ConcurrentBatchIterator
from .cityscapes import Cityscapes


def get_dataset(name, split, folder="/ais/gobi4/mren/data/cityscapes/sem_seg"):
  """Gets a dataset object.

  Args:
    name: String. Name of the dataset.
    split: String. "train", "valid", or "test".
    folder: String. Path where the dataset is stored.

  Returns:
    dataset: A dataset object.
  """
  if name == "cityscapes":
    fname = os.path.join(folder, "{}_full_size.h5".format(split))
    return Cityscapes(fname)
  else:
    raise Exception("Unknown dataset \"{}\"".format(name))


def get_iterator(dataset,
                 batch_size,
                 shuffle=True,
                 cycle=True,
                 log_epoch=-1,
                 multi_worker=False,
                 num_worker=10,
                 queue_size=50,
                 seed=0):
  """Gets a dataset iterator.

  Args:
    dataset: Object. A Dataset object.
    batch_size: Int. Size of the mini-batch.
    shuffle: Bool. Whether to shuffle the data every epoch.
    cycle: Bool. Whether to cycle the dataset or just run for one epoch.
    log_epoch: Int. Number of steps to log the progress. Default -1 with no log.
    multi_worker: Bool. Whether to launch multiple threads for data reading.
    num_worker: Int. Number of background data reading threads.
    queue_size: Int. Maximum number of preprared mini-batch in memory.
    seed: Int. Random seed for data shuffling.
  """
  b = BatchIterator(
      dataset.get_size(),
      batch_size=batch_size,
      log_epoch=log_epoch,
      get_fn=dataset.get_batch,
      cycle=cycle,
      shuffle=shuffle,
      seed=seed)
  if multi_worker:
    b = ConcurrentBatchIterator(
        b, max_queue_size=queue_size, num_threads=num_worker)
  return b


def example():
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
