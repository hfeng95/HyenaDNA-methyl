import torch
from torch.utils.data import Dataset
import pandas as pd

class VeggiesDataset(Dataset):
    def __init__(self,
                 csv_file,
                 transform=None,
                 max_sequence_length=200,
                 cols=['sequence'],
                 tokenizer=None):
        self.transform = transform
        self.seq_len = max_sequence_length
        self.data = pd.read_csv(csv_file)
        self.cols = cols
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sequence = self.data.loc[idx,self.cols[0]]

        seq_tokenized = self.tokenizer(sequence,
            add_special_tokens=False,
            padding='max_length',
            max_length=self.seq_len,
            truncation=True,
        )
        seq_ids = seq_tokenized["input_ids"]  # get input_ids
        seq_ids = torch.tensor(seq_ids)

        return seq_ids[:-1],seq_ids[1:]
