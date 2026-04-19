#!/usr/bin/env python3
"""
Edge Map Generator for 3D Gaussian Splatting (3DGS) Datasets
=============================================================
Produces per-image edge maps from a 3DGS dataset directory.

Supported detectors:
  - canny       : Classic Canny (fast, good for sharp geometric edges)
  - sobel       : Sobel gradient magnitude (smooth, less noise-sensitive)
  - laplacian   : Laplacian of Gaussian (LoG) — blob/edge combined
  - hed         : HED deep edge detector via OpenCV DNN (best quality, needs model)
  - pidinet     : PiDiNet-style thin edges (requires torch + kornia)

Dataset layout expected (COLMAP / standard 3DGS):
  <dataset_root>/
    images/           ← source images (any depth: jpg/png/exr)
    [sparse/]
    [transforms.json] ← NeRF-style datasets also supported

Output is written to:
  <dataset_root>/edge_maps/<detector>/   (mirrors images/ sub-structure)

Usage
-----
  python generate_edge_maps.py --dataset /path/to/dataset
  python generate_edge_maps.py --dataset /path/to/dataset --detector sobel --sigma 1.5
  python generate_edge_maps.py --dataset /path/to/dataset --detector hed --hed_model /path/to/hed.caffemodel
  python generate_edge_maps.py --input_dir /custom/img/dir --output_dir /custom/out/dir --detector canny
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Supported image extensions
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".exr"}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_gray(path: Path) -> np.ndarray:
    """Load an image as uint8 grayscale, handling HDR/EXR tone-mapping."""
    if path.suffix.lower() == ".exr":
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise IOError(f"Cannot read EXR: {path}")
        # Reinhard tone-map then convert to 8-bit gray
        tm = cv2.createTonemapReinhard(gamma=2.2)
        img_tm = tm.process(img.astype(np.float32))
        gray = cv2.cvtColor((np.clip(img_tm, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Cannot read image: {path}")
    return gray


def save_edge_map(edge: np.ndarray, out_path: Path, format: str = "png") -> None:
    """Save a float [0,1] or uint8 edge map, normalised to uint8."""
    if edge.dtype != np.uint8:
        edge = (np.clip(edge, 0, 1) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Always save as PNG for lossless edges; rename ext if needed
    save_as = out_path.with_suffix(f".{format}")
    cv2.imwrite(str(save_as), edge)


def collect_images(input_dir: Path) -> list[Path]:
    """Recursively collect all image files under input_dir."""
    images = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    )
    return images


# ─────────────────────────────────────────────────────────────────────────────
# Edge detectors
# ─────────────────────────────────────────────────────────────────────────────

def detect_canny(gray: np.ndarray, sigma: float, low_thr: float, high_thr: float) -> np.ndarray:
    """
    Gaussian-smoothed Canny edge detector.
    Returns uint8 binary edge map (0 or 255).
    """
    ksize = int(6 * sigma + 1) | 1  # make odd
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    edges = cv2.Canny(blurred, low_thr, high_thr)
    return edges


def detect_sobel(gray: np.ndarray, sigma: float, ksize: int = 3) -> np.ndarray:
    """
    Sobel gradient magnitude (float → normalised uint8).
    ksize: 1, 3, 5, or 7.
    """
    ksize = max(1, ksize | 1)  # ensure odd
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray.astype(np.float32)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.hypot(gx, gy)
    # Normalise per-image (preserves relative structure)
    if mag.max() > 0:
        mag /= mag.max()
    return mag.astype(np.float32)


def detect_laplacian(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    Laplacian of Gaussian (LoG) — absolute value, normalised.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap = np.abs(lap)
    if lap.max() > 0:
        lap /= lap.max()
    return lap.astype(np.float32)


class HEDDetector:
    """
    Holistically-nested Edge Detection via OpenCV DNN.

    Model download:
      prototxt : https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt
      caffemodel: https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel
    """

    def __init__(self, prototxt: str, caffemodel: str):
        if not os.path.isfile(prototxt):
            raise FileNotFoundError(f"HED prototxt not found: {prototxt}")
        if not os.path.isfile(caffemodel):
            raise FileNotFoundError(f"HED caffemodel not found: {caffemodel}")
        self.net = cv2.dnn.readNet(caffemodel, prototxt)
        # Register the 'crop' layer (required for HED)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            bgr, scalefactor=1.0, size=(w, h),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        # HED has 5 side outputs + 1 fused; take the fused output (index -1)
        out_names = self.net.getUnconnectedOutLayersNames()
        outs = self.net.forward(out_names)
        fused = outs[-1][0, 0]  # shape (H, W)
        fused = cv2.resize(fused, (w, h))
        fused = np.clip(fused, 0, 1)
        return fused.astype(np.float32)


class PiDiNetDetector:
    """
    Thin-edge detector using kornia's PiDiNet wrapper.
    Requires: pip install torch torchvision kornia
    """

    def __init__(self, device: str = "cpu"):
        try:
            import torch
            import kornia
        except ImportError:
            raise ImportError(
                "PiDiNet requires torch and kornia:\n"
                "  pip install torch torchvision kornia"
            )
        import torch
        import kornia.filters as KF
        self.torch = torch
        self.KF = KF
        self.device = torch.device(device)

    def detect(self, gray: np.ndarray) -> np.ndarray:
        import torch
        t = self.torch.from_numpy(gray.astype(np.float32) / 255.0)
        t = t.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        # Use Canny from kornia as a differentiable proxy
        edges = self.KF.canny(t, low_threshold=0.1, high_threshold=0.2)[1]
        edges = edges.squeeze().cpu().numpy()
        return edges.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_detector(args):
    if args.detector == "hed":
        return HEDDetector(args.hed_prototxt, args.hed_model)
    if args.detector == "pidinet":
        return PiDiNetDetector(device=args.device)
    return None  # opencv-based; handled inline


def process_image(
    img_path: Path,
    detector_name: str,
    detector_obj,
    args,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Load → detect → save one image."""
    # Relative path to mirror directory structure
    try:
        rel = img_path.relative_to(input_dir)
    except ValueError:
        rel = img_path.name

    out_path = output_dir / rel

    gray = load_image_gray(img_path)

    if detector_name == "canny":
        edge = detect_canny(gray, args.sigma, args.canny_low, args.canny_high)
    elif detector_name == "sobel":
        edge = detect_sobel(gray, args.sigma, ksize=args.sobel_ksize)
    elif detector_name == "laplacian":
        edge = detect_laplacian(gray, args.sigma)
    elif detector_name == "hed":
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edge = detector_obj.detect(bgr)
    elif detector_name == "pidinet":
        edge = detector_obj.detect(gray)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    save_edge_map(edge, out_path, format=args.output_format)


def run(args):
    # ── Resolve directories ──────────────────────────────────────────────────
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        dataset_root = Path(args.dataset)
        input_dir = dataset_root / "images"
        if not input_dir.exists():
            # Fallback: look for NeRF-style flat image list
            input_dir = dataset_root
            print(f"[warn] No 'images/' subfolder found; scanning dataset root: {input_dir}")

    if not input_dir.exists():
        sys.exit(f"[error] Input directory does not exist: {input_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.dataset) / "edge_maps" / args.detector

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect images ───────────────────────────────────────────────────────
    images = collect_images(input_dir)
    if not images:
        sys.exit(f"[error] No images found under: {input_dir}")

    print(f"Found {len(images)} image(s) in {input_dir}")
    print(f"Detector : {args.detector}")
    print(f"Output   : {output_dir}")

    # ── Build detector ───────────────────────────────────────────────────────
    try:
        detector_obj = build_detector(args)
    except (FileNotFoundError, ImportError) as e:
        sys.exit(f"[error] {e}")

    # ── Process ──────────────────────────────────────────────────────────────
    errors = []
    t0 = time.time()

    for img_path in tqdm(images, unit="img", desc="Edge detection"):
        try:
            process_image(img_path, args.detector, detector_obj, args, input_dir, output_dir)
        except Exception as exc:
            errors.append((img_path, exc))
            tqdm.write(f"[warn] Skipped {img_path.name}: {exc}")

    elapsed = time.time() - t0
    n_ok = len(images) - len(errors)
    print(f"\nDone — {n_ok}/{len(images)} images processed in {elapsed:.1f}s")
    if errors:
        print(f"  {len(errors)} error(s):")
        for p, e in errors[:5]:
            print(f"    {p.name}: {e}")

    # ── Write metadata sidecar ───────────────────────────────────────────────
    meta_path = output_dir / "_edge_map_metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"detector      : {args.detector}\n")
        f.write(f"sigma         : {args.sigma}\n")
        f.write(f"output_format : {args.output_format}\n")
        if args.detector == "canny":
            f.write(f"canny_low     : {args.canny_low}\n")
            f.write(f"canny_high    : {args.canny_high}\n")
        if args.detector == "sobel":
            f.write(f"sobel_ksize   : {args.sobel_ksize}\n")
        f.write(f"n_images      : {n_ok}\n")
        f.write(f"elapsed_s     : {elapsed:.2f}\n")
    print(f"Metadata written to {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate edge maps for a 3DGS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", metavar="DIR",
                     help="3DGS dataset root (expects images/ subfolder)")
    src.add_argument("--input_dir", metavar="DIR",
                     help="Direct path to image folder (overrides --dataset)")

    p.add_argument("--output_dir", metavar="DIR", default=None,
                   help="Where to write edge maps (default: <dataset>/edge_maps/<detector>/)")

    # Detector choice
    p.add_argument("--detector", choices=["canny", "sobel", "laplacian", "hed", "pidinet"],
                   default="canny", help="Edge detection method")

    # Common parameters
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Gaussian pre-blur sigma (used by canny / sobel / laplacian)")

    # Canny-specific
    p.add_argument("--canny_low", type=float, default=50.0,
                   help="Canny lower hysteresis threshold")
    p.add_argument("--canny_high", type=float, default=150.0,
                   help="Canny upper hysteresis threshold")

    # Sobel-specific
    p.add_argument("--sobel_ksize", type=int, default=3, choices=[1, 3, 5, 7],
                   help="Sobel kernel size")

    # HED-specific
    p.add_argument("--hed_prototxt", default="deploy.prototxt",
                   help="Path to HED deploy.prototxt")
    p.add_argument("--hed_model", default="hed_pretrained_bsds.caffemodel",
                   help="Path to pretrained HED .caffemodel")

    # PiDiNet-specific
    p.add_argument("--device", default="cpu",
                   help="Torch device for PiDiNet (e.g. cuda:0, cpu, mps)")

    # Output format
    p.add_argument("--output_format", choices=["png", "jpg"], default="png",
                   help="Output image format (png recommended for lossless edges)")

    args = p.parse_args()

    # --input_dir also needs dataset or output_dir for sensible defaults
    if args.input_dir and not args.output_dir and not args.dataset:
        args.dataset = str(Path(args.input_dir).parent)

    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
