import os
import zipfile
import requests
from pathlib import Path
from typing import Tuple, List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    data_path = Path("data/")
    image_path = data_path / destination
    if image_path.is_dir():
        print(f"[INFO] {image_path} already exists, skipping download.")
    else:
        print(f"[INFO] Creating {image_path}...")
        image_path.mkdir(parents=True, exist_ok=True)
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            print(f"[INFO] Downloading {target_file}...")
            response = requests.get(source)
            response.raise_for_status()
            f.write(response.content)
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file}...")
            zip_ref.extractall(image_path)
        if remove_source:
            os.remove(data_path / target_file)
    return image_path


def create_dataloaders(
    train_dir,
    test_dir,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dataloader, test_dataloader, class_names
