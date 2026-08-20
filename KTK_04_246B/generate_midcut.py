"""Generate line-art inbetweens with the arbitrary-timestep RIFEm checkpoint."""
from pathlib import Path
import sys
import argparse

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
RIFE_ROOT = ROOT.parent / "ECCV2022-RIFE-main"
sys.path.insert(0, str(RIFE_ROOT))
from model.RIFE import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)


def line_normalize(frame):
    """Remove interpolation haze while retaining the original black/blue/red line palette."""
    rgb = frame[0].detach().float().cpu().permute(1, 2, 0).numpy()
    # RIFE attenuates thin pencil lines towards white.  Expand the distance from
    # paper white (per channel) before removing only near-white compression haze.
    rgb = 1.0 - np.clip((1.0 - rgb) * 2.0, 0.0, 1.0)
    rgb[rgb.min(axis=2) > 0.985] = 1.0
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def save_selected(model, first_name, last_name, frame_numbers, output, scale, tta):
    first, last = read(ROOT / "origin" / "midcut" / first_name).to(DEVICE), read(ROOT / "origin" / "midcut" / last_name).to(DEVICE)
    _, _, h, w = first.shape
    padding = (0, (-w) % 32, 0, (-h) % 32)
    first, last = F.pad(first, padding), F.pad(last, padding)
    interval_count = len(frame_numbers) + 1
    for offset, number in enumerate(frame_numbers, start=1):
        frame = model.inference(first, last, scale=scale, TTA=tta, timestep=offset / interval_count)
        data = line_normalize(frame[:, :, :h, :w])
        Image.fromarray(data, "RGB").save(output / f"A{number:04d}.tga")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "generated" / "midcut")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    model = Model(arbitrary=True)
    model.load_model(str(RIFE_ROOT / "train_log"), -1)
    model.eval()
    model.device()
    with torch.no_grad():
        save_selected(model, "A0001.tga", "A0006.tga", [2, 3, 4, 5], output, args.scale, args.tta)
        save_selected(model, "A0006.tga", "A0009.tga", [7, 8], output, args.scale, args.tta)
    print(f"Saved six frames to {output}")


if __name__ == "__main__":
    main()
