import numpy as np

def compute_dice_score(predictions, labels):
    """
    Compute Dice score for binary segmentation.
    
    Args:
        predictions (numpy array): Predicted segmentation masks.
        labels (numpy array): Ground truth segmentation masks.
        
    Returns:
        float: Dice score.
    """
    predictions = predictions > 0.5  # Apply threshold to predictions
    labels = labels > 0.5  # Apply threshold to labels
    intersection = np.logical_and(predictions, labels)
    dice_score = 2 * np.sum(intersection) / (np.sum(predictions) + np.sum(labels))
    return dice_score

def visualize_3d_segmentations(prediction, ground_truth):
    """
    Visualize 3D segmentations using matplotlib.
    
    Args:
        prediction (numpy array): Predicted segmentation mask.
        ground_truth (numpy array): Ground truth segmentation mask.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot voxel grids
    ax.voxels(prediction, edgecolor='red', label='Prediction')
    ax.voxels(ground_truth, edgecolor='green', label='Ground Truth')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Segmentations')
    ax.legend()

    plt.show()
