import pandas as pd


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
