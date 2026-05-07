import sys
from train import parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "custom"])
    args = parse_args()
    assert args.model == "custom"
    assert args.epochs == 10
    assert args.batch_size == 32
    assert args.lr == 3e-3
    assert args.device == "cpu"
