import torch
import torch.optim as optim
from model import VNet, DiceLoss
from data_loader import get_dataloader
import os
import numpy as np

num_epochs = 10
batch_size = 4
learning_rate = 1e-4

def train():
    """
    Train the VNet model for 3D segmentation.
    """
    train_loader = get_dataloader('/data/processed/images', '/data/processed/labels', batch_size=batch_size)
    val_loader = get_dataloader('/data/processed/images', '/data/processed/labels', batch_size=batch_size, shuffle=False)

    model = VNet()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        train_loss = 0
        for batch in train_loader:
            images, labels = batch
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0
        dice_scores = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                dice_score = criterion(outputs, labels)
                dice_scores.append(dice_score.item())

        avg_dice_score = np.mean(dice_scores)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, Avg Dice Score: {avg_dice_score:.4f}')

    # Save the model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/model_vnet.pth')

if __name__ == "__main__":
    train()
