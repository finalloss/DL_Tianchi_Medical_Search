import json

import torch
from torch.utils.data import Dataset

class MedicalQueryDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file) as input_data:
            self.json_content = json.load(input_data)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.json_content)

    def __getitem__(self, idx):
        block = self.json_content[idx]
        query1 = block['query1']
        query2 = block['query2']
        label = block['label']
        encoding = self.tokenizer(query1, query2, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(int(label), dtype=torch.long)
        }


