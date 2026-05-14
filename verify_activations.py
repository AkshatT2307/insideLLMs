#!/usr/bin/env python3
"""
verify_activations.py — Sanity-check a saved HDF5 activation file.

Reads the file, prints metadata, checks shapes / dtypes / NaN counts,
and optionally loads a few samples to confirm readability.

Usage
-----
    python verify_activations.py activations/cs_activations.h5
"""

import argparse
import sys

import h5py
import numpy as np

COMPONENTS = ("attn", "mlp", "layer")
MODES = ("last", "mean")


def verify(filepath: str, check_samples: int = 5) -> bool:
    """Run checks on a single HDF5 file. Returns True if all pass."""

    print(f"\n{'='*60}")
    print(f"Verifying: {filepath}")
    print(f"{'='*60}")

    ok = True

    with h5py.File(filepath, "r") as f:
        # ── attributes ───────────────────────────────────────────────────
        print("\nAttributes:")
        for key in ("domain", "model", "num_layers", "hidden_dim", "max_length", "num_samples"):
            val = f.attrs.get(key, "MISSING")
            print(f"  {key:15s}: {val}")
            if val == "MISSING":
                print(f"  ⚠  Attribute '{key}' is missing!")
                ok = False

        N = int(f.attrs["num_samples"])
        L = int(f.attrs["num_layers"])
        d = int(f.attrs["hidden_dim"])

        # ── metadata/indices ─────────────────────────────────────────────
        idx = f["metadata/indices"]
        print(f"\nmetadata/indices: shape={idx.shape}, dtype={idx.dtype}")
        if idx.shape != (N,):
            print(f"  ✗ Expected shape ({N},), got {idx.shape}")
            ok = False
        else:
            print(f"  ✓ Shape OK")

        # ── activation datasets ──────────────────────────────────────────
        print("\nActivation datasets:")
        for comp in COMPONENTS:
            for mode in MODES:
                key = f"{comp}/{mode}"
                if key not in f:
                    print(f"  ✗ {key:15s}: MISSING")
                    ok = False
                    continue

                ds = f[key]
                expected_shape = (N, L, d)
                shape_ok = ds.shape == expected_shape
                dtype_ok = ds.dtype == np.float16

                # Check a small slice for NaNs
                sample_count = min(check_samples, N)
                sample = ds[:sample_count]
                nan_count = np.isnan(sample).sum()
                inf_count = np.isinf(sample).sum()

                status = "✓" if (shape_ok and dtype_ok and nan_count == 0) else "✗"
                print(
                    f"  {status} {key:15s}: "
                    f"shape={ds.shape} ({'OK' if shape_ok else 'EXPECTED ' + str(expected_shape)}), "
                    f"dtype={ds.dtype} ({'OK' if dtype_ok else 'EXPECTED float16'}), "
                    f"NaN={nan_count}, Inf={inf_count} (checked first {sample_count})"
                )
                if not shape_ok or not dtype_ok:
                    ok = False
                if nan_count > 0:
                    print(f"      ⚠  NaN values detected!")
                if inf_count > 0:
                    print(f"      ⚠  Inf values detected!")

        # ── value statistics for first sample ────────────────────────────
        print(f"\nValue statistics (sample 0, layer 0):")
        for comp in COMPONENTS:
            for mode in MODES:
                key = f"{comp}/{mode}"
                if key in f:
                    val = f[key][0, 0, :]  # first sample, first layer
                    print(
                        f"  {key:15s}: "
                        f"min={val.min():.4f}, max={val.max():.4f}, "
                        f"mean={val.mean():.4f}, std={val.std():.4f}, "
                        f"norm={np.linalg.norm(val):.4f}"
                    )

    verdict = "ALL CHECKS PASSED ✓" if ok else "SOME CHECKS FAILED ✗"
    print(f"\n{verdict}\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify HDF5 activation files.")
    parser.add_argument("files", nargs="+", help="Paths to .h5 files to verify.")
    parser.add_argument(
        "--check-samples",
        type=int,
        default=5,
        help="Number of samples to check for NaN/Inf (default: 5).",
    )
    args = parser.parse_args()

    all_ok = True
    for fpath in args.files:
        if not verify(fpath, check_samples=args.check_samples):
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
