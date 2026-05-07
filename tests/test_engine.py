import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from vit.engine import train_step, test_step, train

NUM_CLASSES = 3
BATCH_SIZE = 4
IMG_SIZE = 32


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * IMG_SIZE * IMG_SIZE, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


def _fake_dl():
    X = torch.randn(8, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, NUM_CLASSES, (8,))
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)


def test_train_step_returns_float_loss_and_acc():
    device = torch.device("cpu")
    model = _TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = train_step(model, _fake_dl(), nn.CrossEntropyLoss(), optimizer, device)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_test_step_returns_float_loss_and_acc():
    device = torch.device("cpu")
    model = _TinyModel().to(device)
    loss, acc = test_step(model, _fake_dl(), nn.CrossEntropyLoss(), device)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_train_returns_results_dict_with_correct_keys_and_length():
    device = torch.device("cpu")
    model = _TinyModel().to(device)
    dl = _fake_dl()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    results = train(model, dl, dl, optimizer, nn.CrossEntropyLoss(), epochs=2, device=device)
    assert set(results.keys()) == {"train_loss", "train_acc", "test_loss", "test_acc"}
    assert len(results["train_loss"]) == 2
