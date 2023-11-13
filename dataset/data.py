import os
import random
import re
import pickle
import requests
import torch
from dataset.downloads import downloads
from torch.utils.data import Dataset, random_split
from transformers import RobertaTokenizer


class AuthorshipPairDataset(Dataset):
    """Authorship Pair Dataset

    Args:
        text_data1 (list[str]): OG Author Texts
        text_data2 (list[str]): Unknown Author Text
        labels (list[int]): Same or Different Booleans (1 or 0)
    """

    def __init__(
        self,
        text_data1: list[str],
        text_data2: list[str],
        labels: list[int],
    ) -> None:
        self.text_data1 = text_data1
        self.text_data2 = text_data2
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        max_length = 512
        text1 = self.text_data1[idx]
        text2 = self.text_data2[idx]
        inputs1 = tokenizer(
            text1,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        inputs2 = tokenizer(
            text2,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        label = torch.tensor(self.labels[idx])

        return inputs1, inputs2, label


def download_file(name: str, url: str, start: re.Pattern, end: re.Pattern):
    """Downloads a file and adds it to the database

    Args:
        name (str): file save name
        url (str): download from where?
        start (re.Pattern): Start string of save
        end (re.Pattern): Post-string of save

    Raises:
        requests.RequestException: Raised on failure to download the file
    """
    response = requests.get(url, timeout=1)
    if response.status_code != 200:
        raise requests.RequestException(
            f"Failed to download file from {url} with status code {response.status_code}"
        )
    text = response.content.decode("utf-8")
    text = re.split(start, text, 1)[1]
    text = re.split(end, text, 1)[0]
    text = text.strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_path = os.path.join(script_dir, "database")
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    file_path = os.path.join(database_path, f"{name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


# pylint: disable=too-many-locals
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
    database_dir = "./dataset/database"

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

    pairs = [__generate_pair() for _ in range(size)]
    text_data1, text_data2, labels = zip(*pairs)

    # text_data1 = []
    # text_data2 = []
    # labels = []

    # for _ in range(size):
    #     text1, text2, label = __generate_pair()
    #     text_data1.append(text1)
    #     text_data2.append(text2)
    #     labels.append(label)

    ds = AuthorshipPairDataset(
        text_data1,
        text_data2,
        labels
    )
    train_size = int(train_split * len(ds))
    test_size = len(ds) - train_size

    train_ds, test_ds = random_split(ds, [train_size, test_size])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "train_ds.pkl"), "wb") as f:
        pickle.dump(train_ds, f)
    with open(os.path.join(script_dir, "test_ds.pkl"), "wb") as f:
        pickle.dump(test_ds, f)
    return train_ds, test_ds


def get_data(
    size: int = 10000, train_split: float = 0.8, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Returns training and testing datasets, and updates database if necessary

    Args:
        size (int, optional): Dataset Size. Defaults to 10000.
        train_split (float, optional): Train Test Split. Defaults to 0.8.
        seed (int, optional): Dataset Generation Seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset]: (training dataset, testing dataset)
    """
    local_path = os.path.dirname(os.path.abspath(__file__))
    # Check dataset dir for dataset pkl files, return them if they exist
    train_ds_path = os.path.join(local_path, "train_ds.pkl")
    test_ds_path = os.path.join(local_path, "test_ds.pkl")
    if os.path.exists(train_ds_path) and os.path.exists(test_ds_path):
        with open(train_ds_path, "rb") as f:
            train_ds = pickle.load(f)
        with open(test_ds_path, "rb") as f:
            test_ds = pickle.load(f)
        return train_ds, test_ds

    # Update Database
    db_dir = os.path.join(local_path, "database")
    for author, data in downloads.items():
        if not os.path.exists(os.path.join(db_dir, f"{author}.txt")):
            download_file(author, data[0], data[1], data[2])

    # Return new datasets
    return generate_dataset(size, train_split, seed)
