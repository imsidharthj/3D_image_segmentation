import torch
import torch.nn as nn

class VNet(nn.Module):
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

class DiceLoss(nn.Module):
    """
    Define the Dice loss function for evaluating segmentation performance.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, labels, smooth=1e-6):
        """
        Compute Dice loss.
        
        Args:
            outputs (torch.Tensor): Model outputs.
            labels (torch.Tensor): Ground truth labels.
            smooth (float): Smoothing factor to avoid division by zero.
            
        Returns:
            torch.Tensor: Computed Dice loss.
        """
        assert outputs.shape == labels.shape, f"Output shape {outputs.shape} does not match label shape {labels.shape}"
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation to outputs
        intersection = (outputs * labels).sum()
        dice_score = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
        return 1 - dice_score
