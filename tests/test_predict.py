import sys
from predict import parse_args

def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["predict.py", "--image", "test.jpg", "--model-path", "models/vit.pth", "--model", "custom"])
    args = parse_args()
    assert args.image == "test.jpg"
    assert args.model_path == "models/vit.pth"
    assert args.model == "custom"
    assert args.device == "cpu"
    assert args.plot is False
