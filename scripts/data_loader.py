import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

class NiftiDataset(Dataset):
    """
    Custom Dataset class for loading NIfTI images and labels.
    """
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: (image, label) where image and label are torch tensors.
        """
        image = nib.load(self.image_files[idx]).get_fdata()
        label = nib.load(self.label_files[idx]).get_fdata()

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloader(image_dir, label_dir, batch_size=4, shuffle=True, num_workers=4, transform=None):
    """
    Get DataLoader for the NiftiDataset.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): Directory containing label files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.
        transform (callable, optional): Optional transform to be applied to images.
        
    Returns:
        DataLoader: DataLoader instance for the dataset.
    """
    dataset = NiftiDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
