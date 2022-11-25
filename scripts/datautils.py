import json
import torch
import pytorch_lightning as pl
from typing import Dict, List, Optional, Sequence
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class RRDataset(Dataset):
    def __init__(
        self,
        doc_list: List[Dict[str, any]],
        label2id: Dict[str, int],
        default_label: str = 'NONE'
    ):
        self.doc_list = doc_list
        self.label2id = label2id

        assert default_label in label2id, "Default label has no "\
            "mapping in the `label2id`"

        self.default_label = default_label

        self.build()

    def build(self):
        self.samples = list()

        for doc in self.doc_list:
            doc_samples = doc['annotations'][0]['result']

            sample_idx = 0
            num_samples = len(doc_samples)

            while (sample_idx < num_samples):
                curr_sample = doc_samples[sample_idx]

                # Check if it is the last sample
                if sample_idx == (num_samples - 1):
                    self.samples.append({
                        'text': curr_sample['value']['text'],
                        'label': self.label2id[curr_sample['value']['labels'][0]],
                        'next_label': self.label2id[self.default_label]
                    })

                else:
                    next_sample = doc_samples[sample_idx+1]
                    self.samples.append({
                        'text': curr_sample['value']['text'],
                        'label': self.label2id[curr_sample['value']['labels'][0]],
                        'next_label': self.label2id[next_sample['value']['labels'][0]]
                    })

                sample_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return sample['text'], (sample['label'], sample['next_label'])


class RRBatcher:
    def __init__(self, tnkzr_path: str) -> None:
        """
        Args:
            tnkzr_path (str): Path to load the `Transformers` tokenizer
            to be used.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tnkzr_path)

    def __call__(self, batch: Sequence):
        """Use this function as the `collate_fn` mentioned earlier.
        """
        labels = torch.tensor([sample[1][0] for sample in batch], dtype=torch.int8)
        next_labels = torch.tensor([sample[1][1] for sample in batch], dtype=torch.int8)

        sent_tokens = self.tokenizer(
            [sample[0] for sample in batch],
            max_length=128, # As per https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#fine-tuning
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return sent_tokens, (labels, next_labels)


class RRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batcher: RRBatcher,
        label2id: Dict[str, int],
        batch_size: int = 128
    ):
        """
        Args:
            data_path (str): Path to the pickled annotations file.
            batcher (TripletBatcher): The `Dataloader` to be used. 
        Note:
            `triplets` to be used with `triplet_loss`.
            `twins` to be used with other contrastive loss functions.
        """
        super().__init__()
        self.data_path = data_path

        self.batcher = batcher
        self.batch_size = batch_size
        self.label2id = label2id        

    def setup(self, stage: Optional[str] = None, train_fraction: float = 0.78):
        """Read in the data pickle file and perform splitting here.
        Args:
            train_fraction (float): Fraction to use as training data.
        """
        # Read-in the data
        with open(self.data_path, 'rb') as file:
            full_dataset = json.load(file)

        if stage == "fit" or stage is None:
            # Assign train/val dataset indices for use in dataloaders
            # Only selecting indices to avoid data duplication
            train_indices, val_indices = train_test_split(
                range(len(full_dataset)),
                train_size=train_fraction,
                random_state=44
            )

            # The train dataset
            tr_dataset = [full_dataset[idx] for idx in train_indices]

            self.train_dataset = RRDataset(tr_dataset, self.label2id)

            # The val dataset
            val_dataset = [full_dataset[idx] for idx in val_indices]

            self.val_dataset = RRDataset(val_dataset, self.label2id)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = RRDataset(full_dataset, self.label2id)

        if stage == "predict" or stage is None:
            self.pred_dataset = RRDataset(full_dataset, self.label2id)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.batcher)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.batcher)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.batcher)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, collate_fn=self.batcher)
