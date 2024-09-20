import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from main import VNet  # Ensure this matches your actual model import
import torch


def test(model_path, test_image_path, input_shape, num_classes, device):
    """
    Test the trained V-Net model on a test image and visualize the result.

    Args:
        model_path (str): Path to the trained model weights.
        test_image_path (str): Path to the test NIfTI image.
        input_shape (tuple): Shape of the input image (depth, height, width, channels).
        num_classes (int): Number of segmentation classes.
        device (torch.device): The device (CPU or GPU) to use for inference.

    Returns:
        None
    """

    # Initialize the model and load trained weights
    model = VNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the test image
    image = nib.load(test_image_path).get_fdata()
    image = (image - np.mean(image)) / np.std(image)  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add channel dimension (for grayscale)
    test_image = torch.Tensor(image).unsqueeze(0).to(device)  # Add batch dimension

    # Disable gradients for testing
    with torch.no_grad():
        predicted_mask = model(test_image)
    
    # Apply sigmoid and threshold to get binary predictions (for binary segmentation)
    predicted_mask = torch.sigmoid(predicted_mask)
    predicted_mask = (predicted_mask > 0.5).float()

    # Convert to numpy for visualization
    predicted_mask = predicted_mask.cpu().numpy()

    # Visualize the results
    slice_num = test_image.shape[4] // 2  # Choose the middle slice to display
    plt.figure(figsize=(12, 6))

    # Original image slice
    plt.subplot(1, 2, 1)
    plt.imshow(test_image[0, 0, :, :, slice_num].cpu(), cmap='gray')
    plt.title(f'Original Image - Slice {slice_num}')
    plt.axis('off')

    # Predicted segmentation mask slice
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[0, 0, :, :, slice_num], cmap='gray')
    plt.title(f'Predicted Segmentation - Slice {slice_num}')
    plt.axis('off')

    plt.show()

# Test parameters
input_shape = (512, 512, 129)  # Adjust according to your resampled size
num_classes = 2  # Binary segmentation
model_path = 'saved_models/model_vnet_final.pth'
test_image_path = '/home/sidharth/Workspace/python/3D_image_segmentation/data/test/images/FLARE22_Tr_0047_0000.nii.gz'

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Call the test function
test(model_path, test_image_path, input_shape, num_classes, device)