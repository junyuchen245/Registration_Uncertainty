import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

class ACDCMMDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def norm_img(self, img):
        max_val = np.percentile(img, 99.5)
        min_val = np.percentile(img, 0.5) + 1e-6
        norm_ = (img - min_val) / (max_val - min_val)
        norm_[norm_ > 1] = 1
        norm_[norm_ < 0] = 0
        return norm_

    def convert_lbl(self, lbl):
        lbl_out = np.zeros_like(lbl)
        lbl_out[lbl == 1] = 3
        lbl_out[lbl == 2] = 2
        lbl_out[lbl == 3] = 1
        return lbl_out
    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg, y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        if 'MM_' in path:
            x_seg = self.convert_lbl(x_seg)
            y_seg = self.convert_lbl(y_seg)
        x = self.norm_img(x)
        y = self.norm_img(y)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)