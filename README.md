# ViT Paper Replication

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

PyTorch replication of "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020) applied to FoodVision Mini — a 3-class food image classifier (pizza, steak, sushi).

Includes a from-scratch ViT implementation and pretrained ViT-B/16 fine-tuning, with both CLI scripts and a narrative Jupyter notebook.

## Results

| Model | Test Accuracy | Training Time |
|---|---|---|
| ViT from scratch (10 epochs) | ~72% | ~15 min (GPU) |
| Pretrained ViT-B/16 (5 epochs) | ~95% | ~5 min (GPU) |

*Results after training on FoodVision Mini. Times measured on a single GPU.*

## Architecture

ViT-Base/16 — 12 transformer layers, 768-dim embeddings, 12 attention heads, 196 patches per image.

```
Input image [B, 3, 224, 224]
      │
      ▼ Conv2d patch tokeniser (16×16 stride)
[B, 196, 768]  ← 14×14 patches
      │
      ▼ Prepend class token + positional embedding
[B, 197, 768]
      │
      ▼ 12 × TransformerEncoderBlock
      │     ├─ MSA: LayerNorm → MultiheadAttention → residual
      │     └─ MLP: LayerNorm → Linear→GELU→Dropout→Linear → residual
[B, 197, 768]
      │
      ▼ Extract class token x[:, 0] → LayerNorm → Linear
[B, num_classes]
```

## Quickstart

```bash
# Install
git clone https://github.com/Roopesh-BR/VisionTransformer-Paper-Replication.git
cd VisionTransformer-Paper-Replication
pip install -r requirements.txt
pip install -e .

# Train from scratch (downloads data automatically)
python train.py --model custom --epochs 10 --lr 3e-3

# Fine-tune pretrained ViT-B/16
python train.py --model pretrained --epochs 5 --lr 1e-3

# Run inference
python predict.py --image path/to/image.jpg --model-path models/custom_vit.pth --model custom
```

## Project Structure

```
vit-paper-replication/
├── train.py              # CLI: train custom or pretrained ViT
├── predict.py            # CLI: run inference on an image
├── vit/
│   ├── model.py          # ViT architecture from scratch
│   ├── data.py           # FoodVision Mini download + DataLoaders
│   ├── engine.py         # train/eval loops
│   └── utils.py          # save/load, plotting helpers
├── notebooks/
│   └── vit_paper_replication.ipynb
├── tests/
└── requirements.txt
```

## Reference

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). *An image is worth 16x16 words: Transformers for image recognition at scale*. arXiv:2010.11929.
