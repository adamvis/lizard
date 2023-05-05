import torch


class TextDataset:
    def __init__(self, tokenized_texts, labels=None):
        self.tokenized_texts = tokenized_texts
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_texts.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
