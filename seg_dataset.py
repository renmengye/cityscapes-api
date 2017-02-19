from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import h5py
import logger
import numpy as np


class SegDataset(object):

  def __init__(self, h5_fname):
    self.log = logger.get()
    self.h5_fname = h5_fname
    self.log.info("Reading image IDs")
    self.img_ids = self._read_ids()
    pass

  def _read_ids(self):
    self.log.info(self.h5_fname)
    with h5py.File(self.h5_fname, "r") as h5f:
      idx = h5f.keys()
    return idx

  def get_name(self):
    return "unknown"

  def get_str_id(self, idx):
    return str(idx)

  def get_size(self):
    """Get number of examples."""
    return len(self.img_ids)

  def get_default_timespan(self):
    raise Exception("Not implemented")

  def get_num_semantic_classes(self):
    return 1

  def get_full_size_labels(self, img_ids, timespan=None):
    """Get full sized labels."""
    if timespan is None:
      timespan = self.get_default_timespan()
    with h5py.File(self.h5_fname, "r") as h5f:
      num_ex = len(img_ids)
      y_full = []
      for kk, ii in enumerate(img_ids):
        key = self.get_str_id(ii)
        data_group = h5f[key]
        if "label_ins_seg_full" in data_group:
          y_gt_group = data_group["label_ins_seg_full"]
          num_obj = len(y_gt_group.keys())
          y_full_kk = None
          for jj in range(min(num_obj, timespan)):
            y_full_jj_str = y_gt_group["{:03d}".format(jj)][:]
            y_full_jj = cv2.imdecode(
                y_full_jj_str, cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32)
            if y_full_kk is None:
              y_full_kk = np.zeros(
                  [timespan, y_full_jj.shape[0], y_full_jj.shape[1]])
            y_full_kk[jj] = y_full_jj
          y_full.append(y_full_kk)
        else:
          y_full.append(np.zeros([timespan] + list(data_group["orig_size"][:])))
    return y_full

  def get_batch(self, idx, timespan=None, variables=None):
    """Get a mini-batch."""
    if timespan is None:
      timespan = self.get_default_timespan()
    if variables is None:
      variables = set(["input", "label_sem_seg"])

    with h5py.File(self.h5_fname, "r") as h5f:
      img_ids = [self.img_ids[_idx] for _idx in idx]
      key = self.get_str_id(img_ids[0])
      num_ex = len(idx)
      created_arr = False
      results = {}
      for kk, ii in enumerate(img_ids):
        key = self.get_str_id(ii)
        data_group = h5f[key]
        x_str = data_group["input"][:]
        x = cv2.imdecode(x_str, -1)
        height = x.shape[0]
        width = x.shape[1]
        depth = x.shape[2]
        num_ori_classes = 8
        num_sem_classes = self.get_num_semantic_classes()
        area_sort = None
        if num_sem_classes == 1:
          nc = 1
        else:
          nc = num_sem_classes + 1  # Including background

        if not created_arr:
          if "source" in data_group:
            results["source"] = []
          if "input" in variables:
            results["input"] = np.zeros(
                [num_ex, height, width, depth], dtype=np.float32)
          if "label_ins_seg" in variables:
            results["label_ins_seg"] = np.zeros(
                [num_ex, timespan, height, width], dtype=np.float32)
          if "input_full" in variables:
            if len(idx) > 1:
              raise Exception(("input_full can be only provided in "
                               "batch_size=1 mode."))
            results["input_full"] = None
          if "label_ins_seg_full" in variables:
            if len(idx) > 1:
              raise Exception(("label_ins_seg_full can be only provided in "
                               "batch_size=1 mode."))
            results["label_ins_seg_full"] = None
          if "label_sem_seg" in variables:
            results["label_sem_seg"] = np.zeros(
                [num_ex, height, width, nc], dtype=np.float32)
          if "label_ins_sem_cls" in variables:
            results["label_ins_sem_cls"] = np.zeros(
                [num_ex, timespan, nc], dtype=np.float32)
          if "label_angle" in variables:
            results["label_angle"] = np.zeros(
                [num_ex, height, width, num_ori_classes], dtype=np.float32)
          if "orig_size" in variables:
            results["orig_size"] = np.zeros([num_ex, 2], dtype="int32")
          created_arr = True

        if "input" in variables:
          results["input"][kk] = x.astype(np.float32) / 255

        if "input_full" in variables:
          if "input_full" in data_group:
            x_full_group = data_group["input_full"]
            x_full_str = x_full_group[:]
            x_full = cv2.imdecode(x_full_str, -1).astype(np.float32) / 255
            results["input_full"] = x_full

        if "label_ins_seg" in variables:
          if "label_ins_seg" in data_group:
            y_gt_group = data_group["label_ins_seg"]
            num_obj = len(y_gt_group.keys())
            _y_gt = []
            # If we cannot fit in all the objects,
            # Sort instances such that the largest will be fed.
            for jj in range(num_obj):
              y_gt_str = y_gt_group["{:03d}".format(jj)][:]
              _y_gt.append(cv2.imdecode(y_gt_str, -1).astype(np.float32))
            area = np.array([yy.sum() for yy in _y_gt])
            area_sort = np.argsort(area)[::-1]
            for jj in range(min(num_obj, timespan)):
              results["y_gt"][kk, jj] = _y_gt[area_sort[jj]]

        if "label_ins_seg_full" in variables:
          if "label_ins_seg_full" in data_group:
            y_gt_full_group = data_group["label_ins_seg_full"]
            num_obj = len(y_gt_full_group.keys())
            _y_gt_full = []
            for jj in range(num_obj):
              y_gt_str = y_gt_full_group["{:03d}".format(jj)][:]
              _y_gt_full.append(cv2.imdecode(y_gt_str, -1).astype(np.float32))
            area = np.array([yy.sum() for yy in _y_gt_full])
            area_sort_full = np.argsort(area)[::-1]
            results["label_ins_seg_full"] = np.zeros(
                [timespan, _y_gt_full[0].shape[0], _y_gt_full[0].shape[1]])
            for jj in range(min(num_obj, timespan)):
              results["label_ins_seg_full"][jj] = _y_gt_full[area_sort_full[jj]]
          else:
            if "orig_size" in data_group:
              results["label_ins_seg_full"] = \
                  np.zeros([timespan] +
                           list(data_group["orig_size"][:]))
            else:
              results["label_ins_seg_full"] = \
                  np.zeros(
                      [timespan] +
                  list(data_group["input_full"].shape))

        if "label_sem_seg" in variables:
          if "label_sem_seg" in data_group:
            c_gt_group = data_group["label_sem_seg"]
            if num_sem_classes > 1:
              for jj in range(num_sem_classes):
                if jj == 0:
                  cid = 255  # Background class.
                else:
                  cid = jj - 1  # Other classes.
                cstr = "{:03d}".format(cid)
                if cstr in c_gt_group:
                  c_gt_str = c_gt_group[cstr][:]
                  results["label_sem_seg"][kk, :, :, jj] = cv2.imdecode(
                      c_gt_str, -1).astype(np.float32)
            else:
              c_gt_str = c_gt_group["000"][:]
              results["label_sem_seg"][kk, :, :, 0] = cv2.imdecode(
                  c_gt_str, -1).astype(np.float32)

        if "ins_sem_cls" in variables:
          if "ins_sem_cls" in data_group:
            c_gt_idx = data_group["ins_sem_cls"][:]
            num_obj = len(c_gt_idx)
            if num_obj > 0:
              c_gt_idx = c_gt_idx[area_sort]

            for jj in range(min(num_obj, timespan)):
              cid = c_gt_idx[jj]
              if cid == 255:
                cid2 = 0
              else:
                cid2 = cid + 1
              results["c_gt_idx"][kk, jj, cid2] = 1.0

            if num_obj < timespan:
              for jj in range(num_obj, timespan):
                results["c_gt_idx"][kk, jj:, 0] = 1.0

        if "label_angle" in variables:
          if "label_angle" in data_group:
            d_gt_str = data_group["label_angle"][:]
            d_gt_ = cv2.imdecode(d_gt_str, -1).astype(np.float32)
            for oo in range(num_ori_classes):
              results["label_angle"][kk, :, :, oo] = (
                  d_gt_ == oo).astype(np.float32)

        # For combined datasets, the source of the data example.
        if "source" in data_group:
          results["source"].append(data_group["source"][0])

        if "orig_size" in variables:
          results["orig_size"][kk] = data_group["orig_size"][:]

    return results
