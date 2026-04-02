#!/usr/bin/env python3
"""Export OSNet to ONNX for Horizon BPU conversion.

Usage:
    python scripts/export_osnet_onnx.py                    # default: osnet_x1_0
    python scripts/export_osnet_onnx.py --variant osnet_x0_25  # lightweight

Output:
    osnet_x1_0_128x256.onnx   (input: 1x3x256x128, output: 1x512)

Next step: convert ONNX to .hbm using Horizon Docker toolchain:
    # On machine with Docker + Horizon SDK
    hb_mapper makertbin --model-type onnx \
        --model osnet_x1_0_128x256.onnx \
        --march nash-e \
        --input-type rgb \
        --input-shape 1x3x256x128 \
        --output osnet_x1_0_128x256.hbm
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qp_perception.reid.osnet import build_osnet


def main():
    parser = argparse.ArgumentParser(description="Export OSNet to ONNX")
    parser.add_argument(
        "--variant", default="osnet_x1_0",
        choices=["osnet_x1_0", "osnet_x0_25"],
        help="OSNet variant (default: osnet_x1_0)",
    )
    parser.add_argument("--output", default=None, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    args = parser.parse_args()

    # Crop size: 128x256 (width x height) — standard Re-ID aspect ratio
    crop_w, crop_h = 128, 256

    out_path = args.output or f"{args.variant}_{crop_w}x{crop_h}.onnx"

    print(f"Loading {args.variant}...")
    model = build_osnet(variant=args.variant, device=torch.device("cpu"))
    model.eval()

    # Dummy input: batch=1, channels=3, height=256, width=128
    dummy = torch.randn(1, 3, crop_h, crop_w)

    # Verify PyTorch output
    with torch.no_grad():
        pt_out = model(dummy)
    print(f"PyTorch output shape: {pt_out.shape}, feature_dim={model.feature_dim}")

    # Export to ONNX (use legacy exporter for Windows compatibility)
    print(f"Exporting to {out_path} (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input": {0: "batch"},
            "features": {0: "batch"},
        },
        dynamo=False,
    )

    # Verify ONNX
    import onnx
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path)
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        diff = np.abs(pt_out.numpy() - ort_out).max()
        print(f"ONNX Runtime verify: max diff = {diff:.6f} (should be < 1e-5)")
    except ImportError:
        print("onnxruntime not installed, skipping verification")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done: {out_path} ({size_mb:.1f} MB)")
    print(f"\nNext: convert to .hbm with Horizon toolchain")


if __name__ == "__main__":
    main()
