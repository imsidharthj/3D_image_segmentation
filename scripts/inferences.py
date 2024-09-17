import torch
from model import VNet
import nibabel as nib
import numpy as np

def infer(input_path, output_path):
    """
    Perform inference on a single NIfTI image and save the segmentation result.
    
    Args:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the segmented NIfTI file.
    """
    model = VNet()
    model.load_state_dict(torch.load('saved_models/model_vnet.pth'))
    model.eval()

    image = nib.load(input_path).get_fdata()
    image = torch.tensor(image, dtype=torch.float32)  # Assuming single-channel CT scan (remove unsqueeze(0) if needed)

    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension for model
        output = output.squeeze().numpy()  # Remove batch dimension

        # Option 1: Save segmentation mask as separate NIfTI file
        nib.save(nib.Nifti1Image(output, np.eye(4)), output_path)

        # Option 2: Save segmentation mask as a new channel in the original CT scan (modify if needed)
        original_data = nib.load(input_path).get_fdata()
        original_data = np.concatenate((original_data, output[..., None]), axis=-1)  # Add segmentation as new channel
        nib.save(nib.Nifti1Image(original_data, np.eye(4)), output_path)

if __name__ == "__main__":
    infer('/home/sidharth/Documents/FLARE22Train/images/FLARE22_Tr_0001_0000.nii.gz', 'results/segmentations/sample_segmented.nii.gz')