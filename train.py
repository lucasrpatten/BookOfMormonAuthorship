import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from dataset.data import get_data

from model import SiameseAuthorshipModel

torch.set_grad_enabled(True)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_ds, test_ds = get_data(100, 0.9)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(test_ds, batch_size=4)

model = SiameseAuthorshipModel(roberta_model="roberta-base", similarity_threshold=3.0)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader): # type: ignore
        texts1, texts2, labels = batch
        optimizer.zero_grad()

        labels = torch.Tensor(labels)
        # Forward Pass
        output1 = model(texts1, texts2)
        loss = criterion(output1, labels.float())

        # Backpropogation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(loss.item())
        print(output1, labels)

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_ds)}")  # type: ignore
