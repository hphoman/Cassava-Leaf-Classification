import json
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from PIL import Image
import time
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


def load_metadata(data_dir: str | Path, return_simple:bool = False) -> (tuple[pd.DataFrame, dict, Path]
                                                                        | tuple[pd.DataFrame, dict, Path, dict]):
    """
    Creates a DataFrame for the labels, generates a mapping dictionary for labels, creates a Path object for the image
    directory, and possibly a dictionary of simplified label names.

    Parameters
    ----------
    data_dir: str or Path
        A string or Path object referring to the directory where the data & relevant csv/json files ares located
    return_simple: bool
        Boolean condition to return the simplified disease name. For example, if return_simple = True the method will
        return a dictionary containing items such as "CGM" instead of "Cassava Green Mottle (CGM)."

    Returns
    -------
        Tuple: A tuple containing:
            - labels (pd.DataFrame): A DataFrame containing the image IDs and it's corresponding label
            - label_map (dict): A dictionary containing a mapping from label IDs to the full name of the disease class
            - image_directory (Path): A Path object to the directory where the images were located
            - simplified_labels (dict): if return_simple is True, return a dictionary of simplified labels.
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    csv_path = data_dir / 'train.csv'
    label_path = data_dir / 'label_num_to_disease_map.json'
    image_dir = data_dir / 'train_images'

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found at {csv_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"File not found at {label_path}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found at {image_dir}")

    df = pd.read_csv(csv_path)

    with open(label_path) as f:
        label_map = json.load(f)

    if return_simple:
        simplified_labels = {}
        for values in label_map.items():
            key, label = values[0], values[1]
            label = label.split('(')[-1].split(')')[0]
            key = int(key)
            simplified_labels[key] = label

        return df, label_map, image_dir, simplified_labels

    return df, label_map, image_dir

def warm_image_cache(directory: str | Path) -> None:
    """
    Allows warming of the image cache before training to help prevent storage bottlenecks.

    Parameters
    ----------
        directory: str or Path
            A string or Path object referring to the directory where the images are located
    """
    if isinstance(directory, str):
        directory = Path(directory)
    files = list(directory.rglob("*"))
    files = [p for p in files if p.is_file()]

    total_bytes = sum(p.stat().st_size for p in files)

    print(
        f"Warming cache for {len(files):,} files "
        f"({total_bytes / 1024**3:.2f} GiB)..."
    )

    start = time.perf_counter()

    for i, path in enumerate(files, 1):
        with open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass

        if i % 2000 == 0:
            print(f"{i:,}/{len(files):,} files")

    elapsed = time.perf_counter() - start
    print(f"Cache warm-up completed in {elapsed:.1f}s")

def create_splits(label_df: pd.DataFrame,
                  random_state:int=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    """
    Creates 80%/10%/10% train, validation, and test splits for our images.

    Parameters
    ----------
        label_df: pd.DataFrame
            A dataframe containing the image IDs and respective labels for all images.

    Returns
    -------
        tuple: A tuple containing:
            - training dataframe (pd.DataFrame):
            - validation dataframe (pd.DataFrame):
            - testing dataframe (pd.DataFrame):
    """
    train_df, temp_df = train_test_split(label_df, test_size=0.2,
                                         random_state=random_state, stratify=label_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                       random_state=random_state, stratify=temp_df['label'])

    assert len(train_df) + len(val_df) + len(test_df) == len(label_df)

    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)

    return train_df, val_df, test_df

def create_transformations() -> tuple[v2.Compose, v2.Compose, v2.Compose]:
    """
    Creates train, validation, and test image transformations.

    Returns
    -------
        tuple: A tuple containing:
            - training transformation (v2.Compose): A composition transformation for training images
            - validation transformation (v2.Compose): A composition transformation for validation images
            - testing transformation (v2.Compose): A composition transformation for test images

    """
    train_transformation = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)), # ResNet-18 expected size
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]) # ResNet-18 expected normalization

    val_transformation = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

    test_transformation = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    return train_transformation, val_transformation, test_transformation

class CassavaDataset(Dataset):
    """
    A class representing the Cassava dataset to be used in dataloaders. The class inherits the Dataset class from
    torch.utils.data.

    Parameters
    ----------
        label_df: pd.DataFrame
            A dataframe containing the image IDs and respective labels for all images.
        image_directory: Path
            A Path object referencing the directory containing the images for this dataset.
        transform: v2.Compose
            A composition transformation for the images in this dataset.
        return_path: bool
            A boolean condition that will, if True, return the image name along with the image and its label. This
            parameter is mainly for reporting purposes and shouldn't be used during training/validation.

    """
    def __init__(self, label_df:pd.DataFrame, image_directory:Path, transform:v2.Compose|None=None,
                 return_path = False):
        self.df = label_df.reset_index(drop=True)
        self.image_dir = image_directory
        if transform is not None:
            self.transform = transform
        self.return_path = return_path

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the image, label, and possibly the image name based on the index passed.

        Parameters
        ----------
        idx: int
            The index of the image to return.

        Returns
        -------
        tuple: A tuple containing:
            - image (torch.Tensor): A PyTorch Tensor containing the image after the respective transformation.
            - label (int): The label of the image.
            - image_name (str): If return_path is True, the image name is also returned.
        """
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image_id"])

        image = Image.open(image_path).convert("RGB")
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)

        return image, label