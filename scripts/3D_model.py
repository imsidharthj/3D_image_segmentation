import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU
import SimpleITK as sitk
import nibabel as nib

# Resampling function for NIfTI images
def resample_nifti(input_dir, output_dir, output_size=(512, 512, 129)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(('.nii', '.nii.gz')):
            file_path = os.path.join(input_dir, file)
            image_sitk = sitk.ReadImage(file_path)

            # Calculate new spacing
            original_size = image_sitk.GetSize()
            original_spacing = image_sitk.GetSpacing()
            new_spacing = [
                original_spacing[0] * original_size[0] / output_size[0],
                original_spacing[1] * original_size[1] / output_size[1],
                original_spacing[2] * original_size[2] / output_size[2]
            ]

            # Resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(output_size)
            resampler.SetOutputOrigin(image_sitk.GetOrigin())
            resampler.SetOutputDirection(image_sitk.GetDirection())

            # Execute resampling
            image_resampled_sitk = resampler.Execute(image_sitk)
            output_file = os.path.join(output_dir, file)
            sitk.WriteImage(image_resampled_sitk, output_file)
            print(f"Resampled {file} to size {output_size} and saved to {output_file}")


# Load and preprocess data
def load_and_preprocess_data(image_dir, label_dir, batch_size):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]
    assert len(image_files) == len(label_files), "Image and label counts do not match"

    images = []
    labels = []

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Normalize image and ensure shape is consistent
        image = (image - np.mean(image)) / np.std(image)
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale

        # One-hot encode label and ensure shape
        label = to_categorical(label, num_classes=2)

        images.append(image)
        labels.append(label)

        # Create batches
        if len(images) == batch_size:
            yield np.array(images), np.array(labels)
            images = []
            labels = []

    # Yield the remaining data if not empty
    if len(images) > 0:
        yield np.array(images), np.array(labels)


# Convolution block for V-Net model
def convolution_block(x, num_filters, kernel_size, activation):
    x = Conv3D(num_filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

# V-Net architecture definition
def build_vnet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    c1 = convolution_block(inputs, 16, (3, 3, 3), 'relu')
    p1 = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c1)
    c2 = convolution_block(p1, 32, (3, 3, 3), 'relu')
    p2 = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c2)
    c3 = convolution_block(p2, 64, (3, 3, 3), 'relu')
    p3 = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c3)
    c4 = convolution_block(p3, 128, (3, 3, 3), 'relu')
    u5 = UpSampling3D(size=(2, 2, 2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = convolution_block(u5, 64, (3, 3, 3), 'relu')
    u6 = UpSampling3D(size=(2, 2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = convolution_block(u6, 32, (3, 3, 3), 'relu')
    u7 = UpSampling3D(size=(2, 2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = convolution_block(u7, 16, (3, 3, 3), 'relu')
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Dice coefficient
def dice_coefficient(y_pred, y_true):
    intersection = tf.reduce_sum(y_pred * y_true)
    union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return tf.reduce_mean(dice)

# Training loop for V-Net
def train_vnet(dataloader, num_epochs, learning_rate):
    input_shape = (512, 512, 129, 1)  # Adjust input shape if needed
    num_classes = 2
    model = build_vnet(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy', dice_coefficient])

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    for epoch in range(num_epochs):
        epoch_dice = 0.0
        epoch_loss = 0.0
        steps = 0
        for images, labels in dataloader:
            labels = tf.cast(labels, tf.float32)
            loss, accuracy, dice = model.train_on_batch(images, labels)
            epoch_loss += loss
            epoch_dice += dice
            steps += 1
            print(f"Step {steps}, Loss: {loss:.4f}, Dice: {dice:.4f}")

        avg_loss = epoch_loss / steps
        avg_dice = epoch_dice / steps
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")
        
        # Save model after each epoch
        model.save_weights(f"saved_models/model_vnet_epoch_{epoch+1}.h5")

    # Save final model
    model.save_weights("saved_models/model_vnet_final.h5")
    print("Training complete. Model saved as model_vnet_final.h5.")


# Main paths and setup
image_input_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/raw/images'
label_input_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/raw/labels'
image_output_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/images'
label_output_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/labels'

# Resample images
resample_nifti(image_input_dir, image_output_dir)
resample_nifti(label_input_dir, label_output_dir)

# Load and batch data
batch_size = 4
train_dataset = load_and_preprocess_data(image_output_dir, label_output_dir, batch_size)

# Train model
train_vnet(train_dataset, num_epochs=10, learning_rate=1e-4)



# Load the trained model.
# Preprocess the test images.
# Run inference using the trained model.
# Visualize the predicted segmentation.
# Evaluate the results (if ground truth labels are available).
# Apply the model to real-world CT scans.
# (Optional) Post-process the segmentation mask for refinement.


# # Load trained model
# def load_trained_vnet(model_path, input_shape, num_classes):
#     model = build_vnet(input_shape, num_classes)
#     model.load_weights(model_path)
#     return model

# # Example of loading the final model
# input_shape = (512, 512, 129, 1)  # Adjust based on your input data
# num_classes = 2
# model_path = 'saved_models/model_vnet_final.h5'  # Path to your saved model
# model = load_trained_vnet(model_path, input_shape, num_classes)


# Preprocess a test image (same as during training)
# def preprocess_image(image_path):
#     image = nib.load(image_path).get_fdata()
#     image = (image - np.mean(image)) / np.std(image)
#     image = np.expand_dims(image, axis=-1)  # Add channel dimension
#     return np.array(image)

# # Example of loading a single test image
# test_image_path = '/path/to/test/image.nii.gz'
# test_image = preprocess_image(test_image_path)
# test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension


# # Predict segmentation on test image
# def predict_segmentation(model, test_image):
#     predicted_mask = model.predict(test_image)
#     predicted_mask = np.argmax(predicted_mask, axis=-1)  # Convert from one-hot encoding
#     return predicted_mask

# # Example of running prediction
# predicted_mask = predict_segmentation(model, test_image)


# import matplotlib.pyplot as plt

# # Visualize original test image and predicted segmentation mask
# def visualize_segmentation(test_image, predicted_mask, slice_num):
#     plt.figure(figsize=(12, 6))

#     # Original image slice
#     plt.subplot(1, 2, 1)
#     plt.imshow(test_image[0, :, :, slice_num, 0], cmap='gray')
#     plt.title(f'Original Image - Slice {slice_num}')

#     # Predicted segmentation mask slice
#     plt.subplot(1, 2, 2)
#     plt.imshow(predicted_mask[0, :, :, slice_num], cmap='gray')
#     plt.title(f'Predicted Segmentation - Slice {slice_num}')
    
#     plt.show()

# # Example of visualizing slice 60
# visualize_segmentation(test_image, predicted_mask, slice_num=60)


# def dice_coefficient(y_true, y_pred):
#     y_true_f = np.ravel(y_true)
#     y_pred_f = np.ravel(y_pred)
#     intersection = np.sum(y_true_f * y_pred_f)
#     dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
#     return dice

# # Example evaluation on a test image
# test_label_path = '/path/to/test/label.nii.gz'
# test_label = nib.load(test_label_path).get_fdata()

# # Ensure shapes match for comparison
# predicted_mask_resized = predicted_mask[0]  # Removing batch dimension
# test_label_resized = test_label[:predicted_mask_resized.shape[0], :predicted_mask_resized.shape[1], :predicted_mask_resized.shape[2]]

# # Evaluate dice score
# dice_score = dice_coefficient(test_label_resized, predicted_mask_resized)
# print(f"Dice Coefficient: {dice_score:.4f}")
