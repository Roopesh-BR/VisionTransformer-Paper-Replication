import torch
import pytest
from pathlib import Path
from torch import nn
from vit.utils import save_model, load_model


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_save_and_load_model_preserves_weights(tmp_path):
    model = _SimpleModel()
    original_weight = model.linear.weight.data.clone()
    save_model(model, str(tmp_path), "test_model.pth")
    assert (tmp_path / "test_model.pth").exists()
    loaded_model = _SimpleModel()
    loaded_model = load_model(loaded_model, str(tmp_path / "test_model.pth"), torch.device("cpu"))
    assert torch.allclose(loaded_model.linear.weight.data, original_weight)


def test_save_model_raises_on_bad_extension(tmp_path):
    model = _SimpleModel()
    with pytest.raises(AssertionError):
        save_model(model, str(tmp_path), "model.txt")
