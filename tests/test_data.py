import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from vit.data import create_dataloaders


def _make_fake_dataset(tmp_path: Path):
    for split in ["train", "test"]:
        for cls in ["pizza", "steak", "sushi"]:
            (tmp_path / split / cls).mkdir(parents=True)
            img = Image.fromarray(np.zeros((64, 64, 3), dtype="uint8"))
            img.save(tmp_path / split / cls / "fake.jpg")


def test_create_dataloaders_returns_correct_types(tmp_path):
    _make_fake_dataset(tmp_path)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dl, test_dl, class_names = create_dataloaders(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        transform=transform,
        batch_size=2,
        num_workers=0,
    )
    assert isinstance(train_dl, DataLoader)
    assert isinstance(test_dl, DataLoader)
    assert sorted(class_names) == ["pizza", "steak", "sushi"]


def test_dataloader_batch_shape(tmp_path):
    _make_fake_dataset(tmp_path)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dl, _, _ = create_dataloaders(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        transform=transform,
        batch_size=2,
        num_workers=0,
    )
    X, y = next(iter(train_dl))
    assert X.shape == (2, 3, 224, 224)
    assert y.shape == (2,)
