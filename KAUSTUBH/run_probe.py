"""
run_probe.py — Run linear probing across all layers and components.

For each layer × component, loads a sample of activations, trains a
logistic regression to classify domain, and records test accuracy.

Usage:
    python run_probe.py [--samples-per-domain 2000]
"""
import argparse
import os
import time
import json
import numpy as np

from config import DOMAINS, COMPONENTS, NUM_LAYERS, MODE, SEED, RESULTS_DIR
from loader import load_layer, get_num_samples
from linear_probe import probe_accuracy
from logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Linear probing: domain from activations")
    p.add_argument("--samples-per-domain", type=int, default=2000,
                   help="Samples per domain for probing")
    return p.parse_args()


def main():
    args = parse_args()
    spd = args.samples_per_domain
    out_dir = os.path.join(RESULTS_DIR, "probe")
    os.makedirs(out_dir, exist_ok=True)
    log = setup_logger("run_probe")

    log(f"Linear Probing Experiment")
    log(f"  Samples/domain: {spd}")
    log(f"  Components: {COMPONENTS}")
    log(f"  Layers: 0-{NUM_LAYERS-1}")
    log(f"  Mode: {MODE}")
    log()

    # ── Check sample counts ──
    for d in DOMAINS:
        n = get_num_samples(d)
        if n < spd:
            log(f"  WARNING: {d} has only {n} samples, using all")

    # ── Run probing ──
    # results[component][layer] = accuracy
    results = {comp: np.zeros(NUM_LAYERS) for comp in COMPONENTS}
    per_class_results = {comp: {} for comp in COMPONENTS}

    t0 = time.time()
    total_tasks = NUM_LAYERS * len(COMPONENTS)
    done = 0

    for comp in COMPONENTS:
        for layer_idx in range(NUM_LAYERS):
            # Load data for this layer × component
            X_parts = []
            y_parts = []

            for label, domain in enumerate(DOMAINS):
                n = get_num_samples(domain)
                n_load = min(spd, n)
                sl = slice(0, n_load)

                acts = load_layer(domain, comp, layer_idx, MODE, sl)  # (n_load, d)
                X_parts.append(acts)
                y_parts.append(np.full(n_load, label, dtype=np.int32))

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)

            # Train probe
            result = probe_accuracy(X, y, seed=SEED)
            results[comp][layer_idx] = result["accuracy"]
            per_class_results[comp][layer_idx] = result["per_class_accuracy"]

            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total_tasks - done) if done > 0 else 0

            log(f"  [{done}/{total_tasks}] {comp} L{layer_idx:2d}: "
                f"acc={result['accuracy']:.4f}  "
                f"(ETA: {eta:.0f}s)")

    elapsed = time.time() - t0

    # ── Save results ──
    np.savez(
        os.path.join(out_dir, "probe_accuracy.npz"),
        **{comp: results[comp] for comp in COMPONENTS},
        domains=np.array(DOMAINS),
    )

    # Save detailed JSON
    summary = {
        "samples_per_domain": spd,
        "mode": MODE,
        "elapsed_seconds": round(elapsed, 1),
        "components": {},
    }
    for comp in COMPONENTS:
        accs = results[comp]
        peak = int(np.argmax(accs))
        summary["components"][comp] = {
            "accuracy_per_layer": [round(float(a), 4) for a in accs],
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "peak_layer": peak,
            "peak_accuracy": round(float(accs[peak]), 4),
            "cv": round(float(np.std(accs) / np.mean(accs)), 4) if np.mean(accs) > 0 else 0,
        }

    with open(os.path.join(out_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──
    log(f"\n{'='*60}")
    log(f"Results (samples/domain={spd}, time={elapsed:.0f}s)")
    log(f"{'='*60}")
    log(f"  {'Component':<12s} {'Mean Acc':>10s} {'Peak Layer':>12s} {'Peak Acc':>10s} {'CV':>8s}")
    log(f"  {'-'*52}")
    for comp in COMPONENTS:
        accs = results[comp]
        peak = int(np.argmax(accs))
        cv = float(np.std(accs) / np.mean(accs)) if np.mean(accs) > 0 else 0
        log(f"  {comp:<12s} {np.mean(accs):>10.4f} {f'L{peak}':>12s} "
            f"{accs[peak]:>10.4f} {cv:>8.4f}")

    log(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
