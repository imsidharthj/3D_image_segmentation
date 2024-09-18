import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import os
import numpy as np
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np


def resample_nifti(input_dir, output_dir, output_size=(512, 512, 129)):
    """
    Resample NIfTI images in input directory and save to output directory.

    Args:
    input_dir (str): Directory containing NIfTI image files.
    output_dir (str): Directory to save resampled NIfTI image files.
    output_size (tuple): Desired output size (default: (512, 512, 129)).
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of NIfTI files in input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))]

    for file in files:
        # Read image using SimpleITK
        file_path = os.path.join(input_dir, file)
        image_sitk = sitk.ReadImage(file_path)

        # Resample image
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetReferenceImage(image_sitk)
        resampler.SetOutputSpacing([image_sitk.GetSpacing()[0] * image_sitk.GetSize()[0] / output_size[0],
                                    image_sitk.GetSpacing()[1] * image_sitk.GetSize()[1] / output_size[1],
                                    image_sitk.GetSpacing()[2] * image_sitk.GetSize()[2] / output_size[2]])
        resampler.SetSize(output_size)
        resampler.SetOutputOrigin(image_sitk.GetOrigin())
        resampler.SetOutputDirection(image_sitk.GetDirection())
        image_resampled_sitk = resampler.Execute(image_sitk)

        # Save resampled image
        output_file = os.path.join(output_dir, file)
        sitk.WriteImage(image_resampled_sitk, output_file)

        print(f"Resampled {file} and saved to {output_file}")


image_input_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/raw/images'
image_output_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/images'
label_input_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/raw/labels'
label_output_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/labels'

resample_nifti(image_input_dir, image_output_dir)
resample_nifti(label_input_dir, label_output_dir)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=(4, 4, 4), stride=(2, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        return x


def load_and_batch_data(image_dir, label_dir, batch_size):
    """
    Loads processed resampled CT images and labels, forms batches, and returns them.

    Args:
        image_dir (str): Directory containing processed CT images.
        label_dir (str): Directory containing processed CT labels.
        batch_size (int): Batch size for training.

    Returns:
        list: A list of tuples, where each tuple contains a batch of images and labels.
    """

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]

    assert len(image_files) == len(label_files), "Image and label counts do not match"

    subjects = []
    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        subject = tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path)
        )
        subjects.append(subject)

    # Create a TorchIO Dataset
    dataset = tio.SubjectsDataset(subjects)

    # Create a TorchIO DataLoader
    transforms = tio.Compose([
        tio.RandomAffine(),
        tio.RandomNoise(),
        tio.RandomFlip(),
    ])

    # Apply transforms during dataset initialization
    transformed_dataset = tio.SubjectsDataset(subjects, transform=transforms)

    # Create a PyTorch DataLoader
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_vnet(dataloader, num_epochs, learning_rate, device):
    """
    Trains the V-Net model on the provided data.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): The device (CPU or GPU) to use for training.

    Returns:
        None
    """

    model = VNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_dice = 0.0

        for step, batch_data in enumerate(dataloader):
            images = batch_data['image'][tio.DATA].to(device)
            labels = batch_data['label'][tio.DATA].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Apply sigmoid to logits and threshold at 0.5
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()

            # Accumulate loss
            running_loss += loss.item()

            # Compute Dice coefficient for the batch
            batch_dice = dice_coefficient(labels, predicted)
            epoch_dice += batch_dice

            print(f"Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}")

        # Learning rate scheduler step
        scheduler.step()

        # Average metrics for the epoch
        avg_loss = running_loss / len(dataloader)
        avg_dice = epoch_dice / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f"saved_models/model_vnet_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), "saved_models/model_vnet_final.pth")
    print("Training complete. Model saved as model_vnet_final.pth.")


def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient for a pair of tensors.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        float: Dice coefficient score.
    """
    smooth = 1e-6
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


if __name__ == "__main__":
    image_dir = "/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/images"
    label_dir = "/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/labels"
    batch_size = 10
    num_epochs = 10
    # learning_rate = 0.001
    learning_rate = 1e-4

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = load_and_batch_data(image_dir, label_dir, batch_size)
    train_vnet(dataloader, num_epochs, learning_rate, device)
