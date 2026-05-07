import torch
import pytest
from vit.model import PatchEmbedding

BATCH = 2
C, H, W = 3, 224, 224
PATCH_SIZE = 16
EMBED_DIM = 768
NUM_PATCHES = (H // PATCH_SIZE) ** 2  # 196
SEQ_LEN = NUM_PATCHES + 1             # 197 (patches + class token)


def test_patch_embedding_output_shape():
    model = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)
    x = torch.randn(BATCH, C, H, W)
    out = model(x)
    assert out.shape == (BATCH, NUM_PATCHES, EMBED_DIM), (
        f"Expected {(BATCH, NUM_PATCHES, EMBED_DIM)}, got {out.shape}"
    )


def test_patch_embedding_rejects_non_divisible_image():
    model = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)
    with pytest.raises(AssertionError):
        model(torch.randn(1, 3, 225, 225))
