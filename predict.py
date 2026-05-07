import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image

from vit.model import ViT
from vit.utils import load_model, pred_and_plot_image

CLASS_NAMES = ["pizza", "steak", "sushi"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained ViT")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model", choices=["custom", "pretrained"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def get_model(model_type: str, model_path: str, device: torch.device) -> torch.nn.Module:
    if model_type == "custom":
        model = ViT(num_classes=len(CLASS_NAMES))
    else:
        model = vit_b_16()
        model.heads = nn.Linear(in_features=768, out_features=len(CLASS_NAMES))
    model = load_model(model, checkpoint_path=model_path, device=device)
    return model.to(device)


def predict(image_path: str, model: torch.nn.Module, device: torch.device) -> tuple:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = get_model(args.model, args.model_path, device)
    class_name, confidence = predict(args.image, model, device)
    print(f"Predicted class: {class_name} | Confidence: {confidence:.3f}")
    if args.plot:
        pred_and_plot_image(
            model=model,
            image_path=args.image,
            class_names=CLASS_NAMES,
            device=device,
        )


if __name__ == "__main__":
    main()
