"""Reference-only evaluator for the KTK_04_246B middle-frame task."""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REF = ROOT / "成品" / "中割"


def lines(path):
    image = np.asarray(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (gray < 245).astype(np.uint8)


def score(pred, ref, tolerance=3):
    # Distance to the closest reference/predicted line pixel in pixels.
    d_ref = cv2.distanceTransform(1 - ref, cv2.DIST_L2, 3)
    d_pred = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 3)
    forward = float(d_ref[pred > 0].mean())
    backward = float(d_pred[ref > 0].mean())
    tp_p = int((d_ref[pred > 0] <= tolerance).sum())
    tp_r = int((d_pred[ref > 0] <= tolerance).sum())
    precision = tp_p / max(int(pred.sum()), 1)
    recall = tp_r / max(int(ref.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"chamfer_px": round((forward + backward) / 2, 3), "f1_tol3": round(f1, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=ROOT / "generated" / "midcut")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    pred_dir = args.pred
    result = {}
    for frame in (2, 3, 4, 5, 7, 8):
        result[f"A{frame:04d}"] = score(lines(pred_dir / f"A{frame:04d}.tga"), lines(REF / f"A{frame:04d}.tga"))
    result["mean"] = {
        key: round(float(np.mean([item[key] for name, item in result.items() if name != "mean"])), 4)
        for key in ("chamfer_px", "f1_tol3")
    }
    sequence = [lines(ROOT / "origin" / "midcut" / "A0001.tga")]
    sequence += [lines(pred_dir / f"A{frame:04d}.tga") for frame in (2, 3, 4, 5)]
    sequence += [lines(ROOT / "origin" / "midcut" / "A0006.tga")]
    sequence += [lines(pred_dir / f"A{frame:04d}.tga") for frame in (7, 8)]
    sequence += [lines(ROOT / "origin" / "midcut" / "A0009.tga")]
    centroids = []
    for mask in sequence:
        ys, xs = np.nonzero(mask)
        centroids.append(np.array([xs.mean(), ys.mean()]))
    velocity = np.diff(np.asarray(centroids), axis=0)
    result["sequence"] = {
        "mean_centroid_acceleration_px": round(float(np.linalg.norm(np.diff(velocity, axis=0), axis=1).mean()), 3)
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (args.report or pred_dir.parent / f"{pred_dir.name}_evaluation.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
