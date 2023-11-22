import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset.data import get_data

from model import SiameseAuthorshipModel


def train_loop(
    batch_size: int = 32,
    epochs: int = 15,
    learning_rate: float = 0.01,
    dataset_size: int = 10000,
    train_split: float = 0.8,
):
    torch.set_grad_enabled(True)

    train_ds, test_ds = get_data(dataset_size, train_split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SiameseAuthorshipModel(roberta_model="roberta-base")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_interval = 1
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):  # type: ignore
            texts1, texts2, labels = batch
            optimizer.zero_grad()

            labels = torch.Tensor(labels)
            # Forward Pass
            output1 = model(texts1, texts2)
            loss = criterion(output1 * 100, labels.float() * 100)

            # Backpropogation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(batch_idx, total_loss / ((batch_idx + 1) * batch_size))

        avg_train_loss = total_loss / len(train_ds)  # type: ignore
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
        model.eval()
        val_labels = []
        val_preds = []
        val_total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_texts1, val_texts2, val_true_labels = val_batch
                val_output = model(val_texts1, val_texts2)
                val_preds.extend(val_output.tolist())
                val_labels.extend(val_true_labels)

        val_accuracy = accuracy_score(
            val_labels, [1 if pred >= 0.8 else 0 for pred in val_preds]
        )
        print(f"Validation Accuracy: {val_accuracy}")
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_ds)}")  # type: ignore

train_loop()