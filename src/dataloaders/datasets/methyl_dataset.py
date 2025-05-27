import torch
from torch.utils.data import Dataset
import pandas as pd

class MethylDataset(Dataset):
    def __init__(self,
                 csv_file,
                 transform=None,
                 max_sequence_length=3200,
                 cols=['sequence','meth'],
                 tokenizer=None):
        self.transform = transform
        self.seq_len = max_sequence_length
        self.data = pd.read_csv(csv_file)   # TODO: use fasta instea
        self.cols = cols
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sequence = self.data.loc[idx,self.cols[0]]
        label = self.data.loc[idx,self.cols[1]]

        seq_tokenized = self.tokenizer(sequence,
            add_special_tokens=False,
            padding='max_length',
            max_length=self.seq_len,
            truncation=True,
        )
        seq_ids = seq_tokenized["input_ids"]  # get input_ids
        seq_ids = torch.tensor(seq_ids)
        label_id = torch.tensor([label])

        return seq_ids,label_id
