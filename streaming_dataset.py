import numpy as np
import pickle
import os
from torch.utils.data import Dataset
from torch import tensor

TOP_DATA_DIRNAME = os.path.join('data', 'final')
DATA_DIRNAME = os.path.join(TOP_DATA_DIRNAME, 'data')
DATA_LIST = [os.path.join(DATA_DIRNAME, file) for file in os.listdir(DATA_DIRNAME)]


class StreamingAccentDataset(Dataset):

    def __init__(self):
        with open(os.path.join(TOP_DATA_DIRNAME, 'idx_to_labels'), 'rb') as f:
            self.idx_to_labels = pickle.load(f)
        self.num_classes = len(self.idx_to_labels)
        self.data = DATA_LIST

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as f:
            return pickle.load(f)  # returns audio, mfcc, label_idx
