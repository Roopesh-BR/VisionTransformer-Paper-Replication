import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from vit.data import download_data, create_dataloaders
from vit.engine import train
from vit.model import ViT
from vit.utils import save_model, plot_loss_curves

DATA_URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT on FoodVision Mini")
    parser.add_argument("--model", choices=["custom", "pretrained"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def train_custom(args, device):
    image_path = download_data(source=DATA_URL, destination="pizza_steak_sushi")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dl, test_dl, class_names = create_dataloaders(
        train_dir=image_path / "train",
        test_dir=image_path / "test",
        transform=transform,
        batch_size=args.batch_size,
    )
    model = ViT(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
    )
    save_model(model, target_dir="models", model_name="custom_vit.pth")
    plot_loss_curves(results, save_path="models/custom_vit_loss_curves.png")
    print("[INFO] Model saved to models/custom_vit.pth")


def train_pretrained(args, device):
    image_path = download_data(source=DATA_URL, destination="pizza_steak_sushi")
    weights = ViT_B_16_Weights.DEFAULT
    auto_transforms = weights.transforms()
    train_dl, test_dl, class_names = create_dataloaders(
        train_dir=image_path / "train",
        test_dir=image_path / "test",
        transform=auto_transforms,
        batch_size=args.batch_size,
    )
    model = vit_b_16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.heads = nn.Linear(in_features=768, out_features=len(class_names))
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
    )
    save_model(model, target_dir="models", model_name="pretrained_vit.pth")
    plot_loss_curves(results, save_path="models/pretrained_vit_loss_curves.png")
    print("[INFO] Model saved to models/pretrained_vit.pth")


def main():
    torch.manual_seed(42)
    args = parse_args()
    device = torch.device(args.device)
    if args.model == "custom":
        train_custom(args, device)
    else:
        train_pretrained(args, device)


if __name__ == "__main__":
    main()
