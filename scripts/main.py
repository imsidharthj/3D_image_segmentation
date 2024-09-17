import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import os
import numpy as np
from torch.utils.data import DataLoader


IMAGE_INPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/raw/images'
LABEL_INPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/raw/labels'
IMAGE_OUTPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/processed/images'
LABEL_OUTPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/processed/labels'


def process_and_save_data(image_path, label_path, output_image, output_label, resample=False, resample_to=(1, 1, 1)):
    # Load the image and label
    image = tio.ScalarImage(image_path)
    label = tio.LabelMap(label_path)
    
    # Check if the image and label have the same spatial shape
    if image.spatial_shape != label.spatial_shape:
        raise ValueError(f"Image and label shapes don't match: {image.spatial_shape} vs {label.spatial_shape}")
    
    # Define the transformation (crop or pad the image/label)
    transform = tio.Compose([
        tio.CropOrPad((512, 512, 129))  # Adjust crop/pad size as needed
    ])
    
    # Apply the transformation
    transformed_image = transform(image)
    transformed_label = transform(label)
    
    # Resample if requested
    if resample:
        resample_transform = tio.Resample(resample_to)
        transformed_image = resample_transform(transformed_image)
        transformed_label = resample_transform(transformed_label)
    
    # Ensure the output folder exists
    os.makedirs(output_image, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)
    
    # Save the processed image and label
    image_output_path = os.path.join(output_image, os.path.basename(image_path))
    label_output_path = os.path.join(output_label, os.path.basename(label_path))
    
    transformed_image.save(image_output_path)
    transformed_label.save(label_output_path)
    
    print(f"Processed and saved: {image_output_path} and {label_output_path}")


class VNet(nn.Module):
    # ... (Your V-Net architecture from the previous response)
    def __init__(self):
        super(VNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ... (similar layers for subsequent encoder blocks)

        # Decoder
        self.convt1 = nn.ConvTranspose3d(32, 16, kernel_size=(4, 4, 4), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm3d(16)
        self.relu4 = nn.ReLU(inplace=True)

        # ... (similar layers for subsequent decoder blocks)

        # Final output layer
        self.conv_final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.relu2(self.bn2(self.conv2(x1)))
        x1_pool = self.pool1(x1)

        # ... (similar operations for subsequent encoder blocks)

        # Decoder
        x1_upsample = self.convt1(x1_pool)
        x1_upsample = self.relu3(self.bn3(x1_upsample))
        x1_upsample = self.relu4(self.bn4(self.conv3(x1_upsample)))

        # ... (similar operations for subsequent decoder blocks)

        # Final output
        out = self.conv_final(x1_upsample)
        return out


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

    Returns:
        None
    """

    model = VNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_data in dataloader:
            images = batch_data['image'][tio.DATA].to(device)
            labels = batch_data['label'][tio.DATA].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "saved_models/model_vnet.pth")


if __name__ == "__main__":
    image_files = [f for f in os.listdir(IMAGE_INPUT_FOLDER) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(LABEL_INPUT_FOLDER) if f.endswith('.nii.gz')]

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(IMAGE_INPUT_FOLDER, image_file)
        label_path = os.path.join(LABEL_INPUT_FOLDER, label_file)
        
        process_and_save_data(
            image_path=image_path,
            label_path=label_path,
            output_image=IMAGE_OUTPUT_FOLDER,
            output_label=LABEL_OUTPUT_FOLDER,
            resample=True,
            resample_to=(1, 1, 1)
        )
    image_dir = "/home/sidharth/Workspace/python/3D_Segmentation_model/data/processed/images"
    label_dir = "/home/sidharth/Workspace/python/3D_Segmentation_model/data/processed/labels"
    batch_size = 10
    num_epochs = 10
    learning_rate = 0.001

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = load_and_batch_data(image_dir, label_dir, batch_size)
    train_vnet(dataloader, num_epochs, learning_rate, device)
