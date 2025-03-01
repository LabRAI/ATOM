"""
build_dataloader.py

This module provides functionality for loading and preprocessing sequence-based datasets 
from a CSV file and constructing PyTorch DataLoaders. It supports handling variable-length 
sequences without padding and ensures proper data splitting for training, validation, and testing.

Functions:
- collate_fn_no_pad(batch): Keeps sequences unpadded while batching.
- preprocess_sequences(df): Converts sequence strings into list[int] format.
- split_and_adjust(dataset_sequences, seed): Splits the dataset into 70% train, 15% validation, and 15% test.
- build_loaders(csv_path, batch_size, drop_last, seed): Loads data from CSV, removes duplicates, processes sequences, 
  and returns train, validation, and test DataLoaders.

Classes:
- SequencesDataset: A PyTorch Dataset that stores sequence-label pairs.
"""


import pandas as pd
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

def collate_fn_no_pad(batch):
    batch_seqs = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]
    return batch_seqs, torch.tensor(batch_labels, dtype=torch.long)

def preprocess_sequences(df):
    def convert_to_list(sequence):
        if isinstance(sequence, str):
            return ast.literal_eval(sequence) 
        return sequence
    df["Sequence"] = df["Sequence"].apply(convert_to_list)
    return df

class SequencesDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.loc[idx, "Sequence"]
        lbl = self.df.loc[idx, "Label"]
        if isinstance(seq, str):
            raise TypeError(f"Sequence should be list[int], but received str: {seq}")
        return list(seq), int(lbl)

def split_and_adjust(dataset_sequences , seed):
    train_df, temp_df = train_test_split(dataset_sequences, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df

def build_loaders(
    csv_path="attack_CiteSeer.csv",
    batch_size=16,
    drop_last=True,
    seed = 42,
):
    df = pd.read_csv(csv_path)
    df_unique = df.drop_duplicates(subset="Sequence")
    df = df_unique
    dataset_sequences = df[["Sequence","Label"]].copy()
    dataset_sequences = preprocess_sequences(dataset_sequences)
    dataset_sequences["Label"] = dataset_sequences["Label"].astype(int)
    dataset_sequences = dataset_sequences[dataset_sequences['Sequence'].apply(len) > 1]


    train_df, val_df, test_df = split_and_adjust(dataset_sequences, seed)

    train_dataset = SequencesDataset(train_df)
    val_dataset   = SequencesDataset(val_df)
    test_dataset  = SequencesDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_no_pad,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_no_pad,
        drop_last=drop_last
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_no_pad,
        drop_last=drop_last
    )

    return train_loader, val_loader, test_loader
