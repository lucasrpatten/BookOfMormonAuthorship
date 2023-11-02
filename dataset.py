import os
import random
from re import split
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class AuthorshipPairDataset(Dataset):
    def __init__(
        self, text_data1: list[str], text_data2: list[str], labels: list[int]
    ) -> None:
        self.text_data1 = text_data1
        self.text_data2 = text_data2
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        text1 = self.text_data1[idx]
        text2 = self.text_data2[idx]
        label = self.labels[idx]

        return (text1, text2, label)


def generate_dataset(
    size: int = 10000, train_split: float = 0.8, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Generates dataset pairs for AI

    Args:
        size (int, optional): Dataset Size. Defaults to 10000.
        train_split (float, optional): Train Test Split. Defaults to 0.8.
        seed (int, optional): Generation Seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset]: Training Dataset, Testing Dataset
    """
    random.seed(seed)
    database_dir = "./database"

    # Get all data files
    files = [
        os.path.join(database_dir, f)
        for f in os.listdir(database_dir)
        if os.path.isfile(os.path.join(database_dir, f))
    ]

    def __generate_pair() -> tuple[str, str, int]:
        # if 1 then same else different
        if random.randint(0, 1) == 1:
            # open random file
            with open(random.choice(files), "r", encoding="utf-8") as f:
                space_split = f.read().split()
            split_start_1 = random.randint(0, len(space_split) - 4096)
            text1_len = random.randint(480, 4090)
            text1 = " ".join(space_split[split_start_1 : split_start_1 + text1_len])
            split_start_2 = random.randint(0, len(space_split) - 4096)
            text2_len = random.randint(480, 4090)
            text2 = " ".join(space_split[split_start_2 : split_start_2 + text2_len])
            return text1, text2, 1
        else:
            file1, file2 = random.sample(files, 2)
            with open(file1, "r", encoding="utf-8") as f:
                space_split1 = f.read().split()
            split_start_1 = random.randint(0, len(space_split1) - 4096)
            text1_len = random.randint(480, 4090)
            text1 = " ".join(space_split1[split_start_1 : split_start_1 + text1_len])
            with open(file2, "r", encoding="utf-8") as f:
                space_split2 = f.read().split()
            split_start_2 = random.randint(0, len(space_split2) - 4096)
            text2_len = random.randint(480, 4090)
            text2 = " ".join(space_split2[split_start_2 : split_start_2 + text2_len])
            return text1, text2, 0

    text_data1 = []
    text_data2 = []
    labels = []
    for _ in range(size):
        text1, text2, label = __generate_pair()
        text_data1.append(text1)
        text_data2.append(text2)
        labels.append(label)

    ds = AuthorshipPairDataset(text_data1, text_data2, labels)
    train_size = int(train_split * len(ds))
    test_size = len(ds) - train_size

    train_ds, test_ds = random_split(ds, [train_size, test_size])

    return train_ds, test_ds
