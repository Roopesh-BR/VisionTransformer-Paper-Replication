import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embedding_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % self.patch_size == 0, (
            f"Image size {x.shape[-1]} must be divisible by patch_size {self.patch_size}"
        )
        return self.flatten(self.patcher(x)).permute(0, 2, 1)


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x_norm, key=x_norm, value=x_norm, need_weights=False
        )
        return attn_output + x
