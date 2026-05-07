from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), (
        "model_name must end with '.pt' or '.pth'"
    )
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {save_path}")
    torch.save(obj=model.state_dict(), f=save_path)


def load_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def plot_loss_curves(results: dict, save_path: str = None):
    epochs = range(len(results["train_loss"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.plot(epochs, results["train_loss"], label="train_loss")
    ax1.plot(epochs, results["test_loss"], label="test_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epochs")
    ax1.legend()
    ax2.plot(epochs, results["train_acc"], label="train_accuracy")
    ax2.plot(epochs, results["test_acc"], label="test_accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Loss curves saved to {save_path}")
    plt.show()


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str],
    device: torch.device,
    transform: transforms.Compose = None,
):
    img = Image.open(image_path)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    model.to(device)
    model.eval()
    with torch.inference_mode():
        pred_logits = model(transform(img).unsqueeze(0).to(device))
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_label]} | Prob: {pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()
