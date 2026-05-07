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


from vit.model import MultiHeadSelfAttentionBlock, MLPBlock, TransformerEncoderBlock


def test_msa_block_output_shape():
    model = MultiHeadSelfAttentionBlock(embedding_dim=768, num_heads=12)
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_msa_block_residual_connection():
    model = MultiHeadSelfAttentionBlock(embedding_dim=768, num_heads=12)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    x = torch.zeros(1, SEQ_LEN, EMBED_DIM)
    out = model(x)
    assert torch.allclose(out, x)


def test_mlp_block_output_shape():
    model = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.1)
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    out = model(x)
    assert out.shape == x.shape


def test_mlp_block_residual_connection():
    model = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    x = torch.zeros(1, SEQ_LEN, EMBED_DIM)
    out = model(x)
    assert torch.allclose(out, x)


def test_transformer_encoder_block_output_shape():
    model = TransformerEncoderBlock(
        embedding_dim=768,
        num_heads=12,
        mlp_size=3072,
        mlp_dropout=0.1,
        attn_dropout=0.0,
    )
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    out = model(x)
    assert out.shape == x.shape
