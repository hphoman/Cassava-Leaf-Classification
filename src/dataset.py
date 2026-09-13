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


def load_metadata(data_dir: str) -> tuple[pd.DataFrame, Path, Path]:
    '''

    '''

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

    return df, label_map, image_dir

def warm_image_cache(directory: str) -> None:
    '''

    '''
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

def create_splits(df: pd.DataFrame,
                  random_state:int=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    '''

    '''
    train_df, temp_df = train_test_split(df, test_size=0.2,
                                         random_state=random_state, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                       random_state=random_state, stratify=temp_df['label'])

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)

    return train_df, val_df, test_df

def create_transformations() -> tuple[v2.Compose, v2.Compose, v2.Compose]:
    '''

    '''
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
    def __init__(self, df:pd.DataFrame, image_dir:Path, transform:v2.Compose=None, return_path = False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir/row['image']

        image = Image.open(image_path).convert("RGB")
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)

        return image, label

def main():
    df, label_path, image_dir = load_metadata("data/")
    print(df.head())
    print(df.shape)
    print(label_path)
    print(image_dir)

    train_df, val_df, test_df = create_splits(df)
    train_transformation, val_transformation, test_transformation = create_transformations()

if __name__ == '__main__':
    main()