import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset.data import get_data

from model import TrainingAuthorshipModel

train_ds, test_ds = get_data(100, 0.9)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(test_ds, batch_size=8)

model = TrainingAuthorshipModel()

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (text1, text2, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model((text1, text2))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {total_loss / 100:.4f}"
        )
        total_loss = 0.0

    # validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text1, text2, labels in val_loader:
            outputs = model((text1, text2))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().items()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {accuracy:.2f}%")


torch.save(model.state_dict(), 'trained_model.pth')

