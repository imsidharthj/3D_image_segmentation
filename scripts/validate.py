import torch
from model import VNet, DiceLoss
from data_loader import get_dataloader
import numpy as np

def validate():
    """
    Validate the VNet model and compute the average Dice score.
    """
    val_loader = get_dataloader('/data/processed/images', '/data/processed/labels', batch_size=4, shuffle=False)

    model = VNet()
    model.load_state_dict(torch.load('saved_models/model_vnet.pth'))
    model.eval()

    dice_scores = []
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            outputs = model(images)
            dice_score = DiceLoss()(outputs, labels)
            dice_scores.append(dice_score.item())

    avg_dice_score = np.mean(dice_scores)
    print(f'Average Dice Score: {avg_dice_score:.4f}')

if __name__ == "__main__":
    validate()
