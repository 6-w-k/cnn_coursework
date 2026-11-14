"""Classify a picture using a saved best validation checkpoint."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from torchvision import transforms
from cnn.data import MEAN, STD
from cnn.models import ResNet18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("image")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ResNet18(checkpoint["config"]["variant"]).eval()
    model.load_state_dict(checkpoint["model"])
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    with Image.open(args.image) as picture:
        image = transform(picture.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(image).softmax(1)[0]
    values, indices = probs.topk(3)
    print(json.dumps({"image": args.image, "variant": checkpoint["config"]["variant"],
        "training_smoke_checkpoint": checkpoint["config"]["smoke"],
        "top3": [{"class": checkpoint["classes"][int(i)], "probability": float(v)} for v, i in zip(values, indices)]}, indent=2))


if __name__ == "__main__":
    main()
