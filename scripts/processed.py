import os
import torchio as tio
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

# Define constants
IMAGE_INPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/raw/images'
LABEL_INPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/raw/labels'
IMAGE_OUTPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/processed/images'
LABEL_OUTPUT_FOLDER = '/home/sidharth/Workspace/python/3D_Segmentation_model/data/processed/labels'

# Define transformations
transform = tio.Compose([
    tio.CropOrPad((512, 512, 129)),
    # tio.Resample((1, 1, 1)),
    # tio.RescaleIntensity()
])

def process_file(file_path, output_folder):
    """
    Process a single NIfTI file using TorchIO.
    
    Args:
        file_path (str): Path to the input NIfTI file.
        output_folder (str): Directory to save the processed image.
    """
    # Load the image
    image = tio.ScalarImage(file_path)

    # Apply the transformation
    transformed_image = transform(image)

    # Save the processed image
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    transformed_image.save(output_path)

    print(f"Processed and saved: {file_path}")

def process_pair(image_file_name, label_file_name):
    """
    Process an image and label pair.
    
    Args:
        image_file_name (str): Name of the image file.
        label_file_name (str): Name of the label file.
    """
    image_file_path = os.path.join(IMAGE_INPUT_FOLDER, image_file_name)
    label_file_path = os.path.join(LABEL_INPUT_FOLDER, label_file_name)
    process_file(image_file_path, IMAGE_OUTPUT_FOLDER)
    process_file(label_file_path, LABEL_OUTPUT_FOLDER)

if __name__ == "__main__":
    # Create output folders if they don't exist
    for folder in [IMAGE_OUTPUT_FOLDER, LABEL_OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get lists of image and label files
    image_files = [file_name for file_name in os.listdir(IMAGE_INPUT_FOLDER) if file_name.endswith('.nii.gz')]
    label_files = [file_name for file_name in os.listdir(LABEL_INPUT_FOLDER) if file_name.endswith('.nii.gz')]

    # Process image and label files in parallel
    pairs = zip(image_files, label_files)
    Parallel(n_jobs=4)(delayed(process_pair)(image_file, label_file) for image_file, label_file in pairs)