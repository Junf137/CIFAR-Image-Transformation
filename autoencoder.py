import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


# Custom dataset class
class RotationDataset(Dataset):
    def __init__(self, inputs, targets, transform=None):
        """
        Args:
            inputs (numpy array): Array of input images (N, C, H, W)
            targets (numpy array): Array of target rotated images (N, C, H, W)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_img = self.inputs[idx]
        target_img = self.targets[idx]

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img


# Training function
def train(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = F.mse_loss(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


# Validation function
def validate(model, val_loader, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = F.mse_loss(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)
    return epoch_loss


# Function to visualize results
def visualize_results(model, data_loader, device, num_samples=5):
    """Visualize some input images and their corresponding predictions"""
    model.eval()

    # Get a batch of data
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs[:num_samples].to(device), targets[:num_samples].to(device)

    # Generate predictions
    with torch.no_grad():
        preds = model(inputs)

    # Move tensors to CPU and convert to numpy arrays
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    # Plot results
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))

    for i in range(num_samples):
        # Display input image
        inputs_img = np.transpose(inputs[i], (1, 2, 0))
        axes[i, 0].imshow(inputs_img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        # Display target image
        targets_img = np.transpose(targets[i], (1, 2, 0))
        axes[i, 1].imshow(targets_img)
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")

        # Display predicted image
        preds_img = np.transpose(preds[i], (1, 2, 0))
        axes[i, 2].imshow(preds_img)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


# Main execution function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 100

    # Load dataset
    # This is a placeholder - you need to load your actual dataset
    # X_train = np.load('X_train.npy')  # Shape: (N, 3, 32, 32)
    # Y_train = np.load('Y_train.npy')  # Shape: (N, 3, 32, 32)

    # Simulate dataset for demonstration
    # Replace this with your actual data loading code
    N = 1000  # Number of samples
    X_data = np.random.rand(N, 3, 32, 32).astype(np.float32)
    Y_data = np.random.rand(N, 3, 32, 32).astype(np.float32)

    # Split into train and validation sets (80-20 split)
    train_size = int(0.8 * len(X_data))
    val_size = len(X_data) - train_size

    X_train, X_val = X_data[:train_size], X_data[train_size:]
    Y_train, Y_val = Y_data[:train_size], Y_data[train_size:]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Create datasets and data loaders
    train_dataset = RotationDataset(X_train, Y_train)
    val_dataset = RotationDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = UNet().to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_loss = float("inf")
    patience = 10  # Early stopping patience
    counter = 0  # Counter for early stopping

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch results
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Model saved!")
            counter = 0  # Reset counter
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping!")
                break

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")

    # Load best model for visualization
    model.load_state_dict(torch.load("best_unet_model.pth"))

    # Visualize some results
    visualize_results(model, val_loader, device)

    # Calculate final MSE on validation set
    final_val_loss = validate(model, val_loader, device)
    final_grade = 100 - 1000 * final_val_loss
    print(f"Final MSE on validation set: {final_val_loss:.6f}")
    print(f"Final Grade: {final_grade:.2f}")

    return model


if __name__ == "__main__":
    main()
