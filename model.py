import torch
import torch.nn as nn


# Define the double convolution block used in U-Net
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder blocks (downsampling)
        self.enc1 = DoubleConvBlock(3, 64)
        self.enc2 = DoubleConvBlock(64, 128)
        self.enc3 = DoubleConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(256, 512)

        # Decoder blocks (upsampling)
        self.dec3 = DoubleConvBlock(512 + 256, 256)
        self.dec2 = DoubleConvBlock(256 + 128, 128)
        self.dec1 = DoubleConvBlock(128 + 64, 64)

        # Final output layer
        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Sigmoid activation for output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        enc1_features = self.enc1(x)
        enc1_pooled = self.pool(enc1_features)

        enc2_features = self.enc2(enc1_pooled)
        enc2_pooled = self.pool(enc2_features)

        enc3_features = self.enc3(enc2_pooled)
        enc3_pooled = self.pool(enc3_features)

        # Bottleneck
        bottleneck_features = self.bottleneck(enc3_pooled)

        # Decoder path with skip connections
        up3 = self.upsample(bottleneck_features)
        concat3 = torch.cat([up3, enc3_features], dim=1)
        dec3_features = self.dec3(concat3)

        up2 = self.upsample(dec3_features)
        concat2 = torch.cat([up2, enc2_features], dim=1)
        dec2_features = self.dec2(concat2)

        up1 = self.upsample(dec2_features)
        concat1 = torch.cat([up1, enc1_features], dim=1)
        dec1_features = self.dec1(concat1)

        # Output layer
        output = self.sigmoid(self.output_conv(dec1_features))

        return output
