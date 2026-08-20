"""Generate KTK_05_140 A/B inbetweens without compositing the layers."""
from pathlib import Path
import sys

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
    image = Image.open(path)
    had_alpha = image.mode == "RGBA"
    if had_alpha:
        # RIFE needs a defined background. Composite only for inference; alpha is
        # reconstructed from the generated line strength on output.
        canvas = Image.new("RGBA", image.size, "white")
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    else:
        image = image.convert("RGB")
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return tensor, had_alpha


def save(frame, path, transparent):
    rgb = frame[0].detach().cpu().permute(1, 2, 0).numpy()
    rgb = 1.0 - np.clip((1.0 - rgb) * 2.0, 0, 1)
    rgb[rgb.min(axis=2) > 0.985] = 1
    output = (rgb * 255).round().astype(np.uint8)
    if transparent:
        alpha = np.where(output.min(axis=2) < 245, 255, 0).astype(np.uint8)
        Image.fromarray(np.dstack((output, alpha)), "RGBA").save(path)
    else:
        Image.fromarray(output, "RGB").save(path)


def generate(model, left_path, right_path, targets, output, prefix, tta=False):
    left, alpha_left = read(left_path)
    right, alpha_right = read(right_path)
    _, _, height, width = left.shape
    padding = (0, (-width) % 32, 0, (-height) % 32)
    left, right = F.pad(left, padding).to(DEVICE), F.pad(right, padding).to(DEVICE)
    count = len(targets) + 1
    for offset, frame_id in enumerate(targets, 1):
        frame = model.inference(left, right, TTA=tta, timestep=offset / count)
        save(frame[:, :, :height, :width], output / f"{prefix}{frame_id:04d}.tga", alpha_left or alpha_right)


def main():
    output_a = ROOT / "generated" / "midcut" / "A"
    output_b = ROOT / "generated" / "midcut" / "B"
    output_a.mkdir(parents=True, exist_ok=True)
    output_b.mkdir(parents=True, exist_ok=True)
    model = Model(arbitrary=True)
    model.load_model(str(RIFE_ROOT / "train_log"), -1)
    model.eval()
    model.device()
    with torch.no_grad():
        generate(model, ROOT / "源文件" / "中割" / "A" / "A0001.tga", ROOT / "源文件" / "中割" / "A" / "A0005.tga", [2, 3, 4], output_a, "A")
        generate(model, ROOT / "成品" / "描原" / "B" / "B0001.tga", ROOT / "成品" / "描原" / "B" / "B0003.tga", [2], output_b, "B", tta=True)
    print(f"Saved A and B layers to {ROOT / 'generated' / 'midcut'}")


if __name__ == "__main__":
    main()
