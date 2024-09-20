import numpy as np
import os
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Dropout, Concatenate, Input, Conv3DTranspose, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for GPU: {gpu}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
else:
    print("Runnig on CPU")


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


def data_load(image_dir, label_dir, batch_size):
    image_files = tf.io.gfile.glob(image_dir + '/*.nii.gz')
    label_files = tf.io.gfile.glob(label_dir + '/*.nii.gz')
    assert len(image_files) == len(label_files), "Image and label counts do not match"

    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))

    def load_and_preprocess(image_file, label_file):
        image_nifti = nib.load(image_file)
        image = image_nifti.get_fdata()
        image = tf.convert_to_tensor(image, tf.float32)
        image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)
        image = tf.expand_dims(image, axis=-1)  # Add channel dimension
        
        label_nifti = nib.load(label_file)
        label = label_nifti.get_fdata()
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        label = tf.one_hot(label, depth=2)
        return image, label
    def tf_load_and_preprocess(image_file, label_file):
        image, label = tf.numpy_function(load_and_preprocess, [image_file, label_file], [tf.float32, tf.float32])
        return image, label

    dataset = dataset.map(tf_load_and_preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def normalization_layer(inputs, mode='batch_norm'):
    normalizations = {
        'batch_norm': layers.BatchNormalization(),
        # Add more normalization methods here
    }
    
    if mode in normalizations:
        return normalizations[mode](inputs)
    else:
        raise ValueError(f"Unsupported normalization: {mode}")


def activation_layer(inputs, activation):
    """Applies the specified activation function to inputs."""
    activations = {
        'relu': layers.ReLU(),          # ReLU doesn't require additional arguments
        'sigmoid': layers.Activation('sigmoid'),  # Directly pass the activation function
    }
    # Apply the activation function to the inputs
    return activations[activation](inputs)



def convolution_layer(inputs, filters, kernel_size, normalization, activation):
    """
    Convolutional layer with optional normalization, activation, and dropout.

    Args:
        inputs (tensor): Input tensor.
        filters (int): Number of filters.
        kernel_size (tuple): Kernel size.
        normalization (bool): Whether to apply normalization.
        activation (str): Activation function.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        tensor: Output tensor.
    """
    x = layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1, 1),
        padding='same',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )(inputs)
    
    x = normalization_layer(x, normalization)
    x = activation_layer(x, activation)
    
    return x


def upsample_layer(inputs, filters, upsampling, activation, mode):
    if upsampling == 'transposed_conv':
        x = tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=2, strides=2, padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        if mode == 'train':
            x = tf.keras.layers.BatchNormalization()(x)

        x = activation_layer(x, activation)

        return x

    else:
        raise ValueError('Unsupported upsampling: {}'.format(upsampling))
    

def concatenate_u_to_c(u, c, kernel_size, depth, normalization, activation, mode):
    """
    Concatenates the upsampling layer (u) to the contracting layer (c) 
    and applies residual block processing, ensuring spatial shapes match.

    Args:
        u (tensor): Upsampling layer output.
        c (tensor): Contracting layer output.
        kernel_size (int): Kernel size for convolutional layers.
        depth (int): Number of convolutional layers in the residual block.
        normalization (str): Normalization method.
        activation (str): Activation function.
        mode (tf.estimator.ModeKeys): Training mode.

    Returns:
        tensor: Output of the concatenated and processed layers.
    """
    with tf.name_scope('concatenate_u_to_c'):
        # Ensure shapes match by cropping the larger tensor
        u_shape = tf.shape(u)
        c_shape = tf.shape(c)

        # Find the minimum depth (3rd dimension)
        min_depth = tf.minimum(u_shape[3], c_shape[3])

        # Crop the larger tensor to match the smaller tensor
        if u_shape[3] > min_depth:
            u = u[:, :, :, :min_depth, :]
        elif c_shape[3] > min_depth:
            c = c[:, :, :, :min_depth, :]

        # Concatenate the u and c layers along the last axis (channel dimension)
        x = tf.concat([u, c], axis=-1)

        # Apply residual block processing
        n_input_channels = x.get_shape()[-1]
        for i in range(depth):
            x = convolution_layer(inputs=x,
                                  filters=n_input_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  normalization=normalization,
                                  activation=activation,
                                  mode=mode)

        return x + tf.identity(x)  # Residual connection



def build_vnet(input_shape, num_classes, dropout_rate=0.2):
    inputs = tf.keras.layers.Input(shape=input_shape) # architectural decision, layer for shape of 3D image

    c1 = convolution_layer(inputs, 16, (3, 3, 3), 'batch_norm', 'relu')
    c1 = layers.Dropout(dropout_rate)(c1)
    p1 = layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c1)
    
    c2 = convolution_layer(p1, 32, (3, 3, 3), 'batch_norm', 'relu')
    c2 = layers.Dropout(dropout_rate)(c2)
    p2 = layers.Conv3D(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c2)
    
    c3 = convolution_layer(p2, 64, (3, 3, 3), 'batch_norm', 'relu')
    c3 = layers.Dropout(dropout_rate)(c3)
    p3 = layers.Conv3D(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c3)
    
    c4 = convolution_layer(p3, 128, (3, 3, 3), 'batch_norm', 'relu')
    
    u5 = upsample_layer(c4, 64, 'transposed_conv', 'relu', mode='train')
    u5 = concatenate_u_to_c(u5, c3, kernel_size=3, depth=2, normalization='batch_norm', activation='relu', mode='train')
    print(f"u5 shape: {u5.shape}")
    u5 = convolution_layer(u5, 64, (3, 3, 3), 'batch_norm', 'relu')
    u5 = layers.Dropout(dropout_rate)(u5)
    
    u6 = upsample_layer(u5, 32, 'transposed_conv', 'relu', mode='train')
    u6 = concatenate_u_to_c(u6, c2, kernel_size=3, depth=2, normalization='batch_norm', activation='relu', mode='train')
    print(f"u6 shape: {u6.shape}")
    u6 = convolution_layer(u6, 32, (3, 3, 3), 'batch_norm', 'relu')
    u6 = layers.Dropout(dropout_rate)(u6)
    
    u7 = upsample_layer(u6, 16, 'transposed_conv', 'relu', mode='train')
    u7 = concatenate_u_to_c(u7, c1, kernel_size=3, depth=2, normalization='batch_norm', activation='relu', mode='train')
    print(f"u7 shape: {u7.shape}")
    u7 = convolution_layer(u7, 16, (3, 3, 3), 'batch_norm', 'relu')
    u7 = layers.Dropout(dropout_rate)(u7)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(u7) # This layer outputs the final predictions. The number of filters is equal to num_classes, and a (1, 1, 1) kernel size means it applies a pointwise convolution.
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', dice_coefficient])
    return model


def dice_coefficient(y_pred, y_true):
    # intersection = tf.reduce_sum(y_pred * y_true)
    # union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    # dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    dice_coefficient = tf.keras.metrics.MeanIoU(num_classes=2)
    dice = dice_coefficient(y_pred, y_true)
    return dice


def train_vnet(dataloader, num_epochs, learning_rate):
    input_shape = (512, 512, 129, 1)
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


base_dir = os.path.dirname(os.path.abspath(__file__))
image_input_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/raw/images'
label_input_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/raw/labels'
image_output_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/images'
label_output_dir = '/home/sidharth/Workspace/python/3D_image_segmentation/data/processed/labels'


resample_nifti(image_input_dir, image_output_dir)
resample_nifti(label_input_dir, label_output_dir)
batch_size = 4
dataset = data_load(image_output_dir, label_output_dir, batch_size)
train_vnet(dataset, num_epochs=10, learning_rate=0.001)


def test(model_path, test_image_path, input_shape, num_classes):
    """
    Test the trained V-Net model on a test image and visualize the result.

    Args:
        model_path (str): Path to the trained model weights.
        test_image_path (str): Path to the test NIfTI image.
        input_shape (tuple): Shape of the input image (depth, height, width, channels).
        num_classes (int): Number of segmentation classes.

    Returns:
        None
    """

    # Initialize the model and load trained weights
    model = build_vnet(input_shape, num_classes)
    model.load_weights(model_path)

    # Load and preprocess the test image
    image = nib.load(test_image_path).get_fdata()
    image = (image - np.mean(image)) / np.std(image)  # Normalize image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (for grayscale)
    test_image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predicted_mask = model.predict(test_image)
    
    # Get binary predictions (for binary segmentation)
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)

    # Visualize the results
    slice_num = test_image.shape[1] // 2  # Choose the middle slice to display
    plt.figure(figsize=(12, 6))

    # Original image slice
    plt.subplot(1, 2, 1)
    plt.imshow(test_image[0, slice_num, :, :, 0], cmap='gray')
    plt.title(f'Original Image - Slice {slice_num}')
    plt.axis('off')

    # Predicted segmentation mask slice
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[0, slice_num, :, :, 0], cmap='gray')
    plt.title(f'Predicted Segmentation - Slice {slice_num}')
    plt.axis('off')

    plt.show()

input_shape = (512, 512, 129, 1)
num_classes = 2
model_path = 'saved_models/model_vnet_final.h5'
test_image_path = '/home/sidharth/Workspace/python/3D_image_segmentation/data/test/images/FLARE22_Tr_0047_0000.nii.gz'

test(model_path, test_image_path, input_shape, num_classes)

