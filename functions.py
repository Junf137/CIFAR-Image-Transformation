import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_images_from_csv(csv_file_path):
    """
    Loads a CSV file, converts pixel vectors back to images, and returns a list of images.

    Args:
        csv_file_path: The path to the CSV file.

    Returns:
        A list of reshaped images. (3 x 32 x 32)
    """
    try:
        df = pd.read_csv(csv_file_path)
        images = []
        for index, row in df.iloc[:, 1:].iterrows():  # Exclude the 'ID' column
            pixel_vector = row.values
            image = pixel_vector.reshape(3, 32, 32)  # Reshape to original CIFAR image size
            images.append(image)
        return images
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None


def prepare_submission(test_data_csv, output_csv, model, batch_size, device):
    """
    Generate predictions for test data and create a submission file.
    """

    # Load the test data
    print(f"Loading test data from {test_data_csv}...")
    X_test = get_images_from_csv(test_data_csv)
    X_test = torch.tensor(np.stack(X_test, axis=0), dtype=torch.float32)
    print(f"Shape of X_test: {X_test.shape}")

    num_samples = X_test.shape[0]

    # Process test data in batches
    print("Generating predictions...")
    Y_pred = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            batch = X_test[i : i + batch_size].to(device)
            outputs = model(batch)
            Y_pred.append(outputs.cpu().numpy())

    # Concatenate all batch predictions
    Y_pred = np.vstack(Y_pred)
    Y_pred_flat = Y_pred.reshape(num_samples, -1)
    len_row = Y_pred_flat.shape[1]

    # Create the DataFrame all at once, which is much more efficient
    pixel_df = pd.DataFrame(Y_pred_flat, columns=[f"p{i}" for i in range(1, len_row + 1)])
    id_df = pd.DataFrame({"ID": list(range(1, num_samples + 1))})

    # Concatenate ID column with pixel data horizontally
    output_df = pd.concat([id_df, pixel_df], axis=1)

    # Save to CSV
    print(f"Saving predictions to {output_csv}...")
    output_df.to_csv(output_csv, index=False)
    print("Done!")


def is_rotated_by_90_or_180(x, y, display=False):
    """
    Check if the image x is rotated by 90 or 180 degrees to image y.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    rotated_90 = transforms.functional.rotate(x_tensor, angle=90)
    rotated_180 = transforms.functional.rotate(x_tensor, angle=180)

    # Display the input image and the rotated images
    if display:
        plt.subplot(1, 2, 1)
        plt.imshow(x.transpose(1, 2, 0))
        plt.title("Input Image")

        plt.subplot(1, 2, 2)
        plt.imshow(y.transpose(1, 2, 0))
        plt.title("Output Image")

        plt.show()

    if torch.all(torch.eq(rotated_90, y_tensor)):
        return 90
    elif torch.all(torch.eq(rotated_180, y_tensor)):
        return 180
    else:
        print("The input image is not rotated by 90 or 180 degrees to the output image.")
        return None


def save_rotation_deg(train_img_x, train_img_y, path):
    # Save the rotation_degrees to a file
    rotation_deg = torch.tensor([])

    for i in tqdm(iterable=range(len(train_img_x)), desc="Checking rotation degrees"):
        rotated_degree = is_rotated_by_90_or_180(train_img_x[i], train_img_y[i], False)

        if rotated_degree is not None:
            rotation_deg = torch.cat((rotation_deg, torch.tensor([rotated_degree])))

    # save the rotation_deg list to a file
    if path is not None:
        torch.save(rotation_deg, path)
