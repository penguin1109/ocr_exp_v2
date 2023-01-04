import torch
from torch.utils import data


class HENDataset(data.Dataset):
    def __init__(self, csv_file_path: str, mode: str):
        super(HENDataset, self).__init__()
        self.mode = mode
        self.csv_file_path = csv_file_path
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass