import pickle
import numpy as np
from torch.utils.data import Dataset


class AccentDataset(Dataset):

    def __init__(self, data_filename, labels_filename):
        with open(data_filename, 'rb') as f:
            self.data = pickle.load(f)
        with open(labels_filename, 'rb') as f:
            self.idx_to_labels = pickle.load(f)
        self.num_classes = len(self.idx_to_labels)
        self.xvar = np.var(np.array([x[0].numpy() for x in self.data]))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
