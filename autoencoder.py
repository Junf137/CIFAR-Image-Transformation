import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from model import UNet
from functions import get_images_from_csv, prepare_submission


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
    plt.savefig("output/predictions.png")
    plt.show()


# Main execution function
def main():

    train_input_csv = "data/train_dataset_input_images.csv"
    train_output_csv = "data/train_dataset_output_images.csv"
    test_input_csv = "data/test_dataset_input_images.csv"
    output_csv = "output/submission.csv"
    checkpoint_path = "output/checkpoints/best_unet_model.pth"

    # Set random seed for reproducibility
    seed = 42
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    # generate wandb run wandb_id, to be used to link the run with test_upload
    wandb_id = wandb.util.generate_id()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 200

    # Initialize model
    model = UNet().to(device)
    summary(model, (3, 32, 32))

    # Load data
    X_train = get_images_from_csv(train_input_csv)
    X_train = np.stack(X_train, axis=0)
    print(f"Shape of X_train: {X_train.shape}")

    Y_train = get_images_from_csv(train_output_csv)
    Y_train = np.stack(Y_train, axis=0)
    print(f"Shape of Y_train: {Y_train.shape}")

    # Split into train and validation sets (80-20 split)
    train_size = int(0.8 * len(X_train))

    # Randomly data split
    indices = np.random.permutation(len(X_train))
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    X_train, X_val = X_train[train_indices], X_train[val_indices]
    Y_train, Y_val = Y_train[train_indices], Y_train[val_indices]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Create datasets and data loaders
    train_dataset = RotationDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = RotationDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            torch.save(model.state_dict(), checkpoint_path)
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
    model.load_state_dict(torch.load(checkpoint_path))

    # Visualize some results
    visualize_results(model, val_loader, device)

    # Calculate final MSE on validation set
    final_val_loss = validate(model, val_loader, device)
    final_grade = 100 - 1000 * final_val_loss
    print(f"Final MSE on validation set: {final_val_loss:.6f}")
    print(f"Final Grade: {final_grade:.2f}")

    # submission
    prepare_submission(test_input_csv, output_csv, model, batch_size, device)

    return model


if __name__ == "__main__":
    main()
