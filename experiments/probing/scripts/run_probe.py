"""
run_probe.py — Run linear probing across all layers and components.

For each layer × component, loads a sample of activations, trains a
logistic regression to classify domain, and records test accuracy.

Usage:
    python run_probe.py                              # uses config.yaml
    python run_probe.py --config ../config.yaml      # explicit config
    python run_probe.py --samples-per-domain 2000
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.logger import setup_logger
from utils.model_loader import get_model_slug

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from loader import ActivationLoader
from linear_probe import probe_accuracy


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Linear probing: domain from activations")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    p.add_argument("--samples-per-domain", type=int, default=None,
                   help="Override samples per domain from config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_slug = get_model_slug(cfg["model"]["name"])
    results_base = os.path.join(os.path.dirname(args.config), cfg.get("results_dir", "./results"))
    results_dir = os.path.join(results_base, model_slug)

    domains = cfg["data"]["domains"]
    components = cfg["extraction"]["components"]
    mode = cfg["extraction"]["pooling"]
    seed = cfg["data"]["seed"]
    spd = args.samples_per_domain or cfg.get("probing", {}).get("samples_per_domain", 2000)

    loader = ActivationLoader(results_dir, pooling=mode)

    # Auto-detect num_layers from activations
    import h5py
    with h5py.File(loader.h5_path(domains[0]), "r") as f:
        num_layers = f[f"{components[0]}/{mode}"].shape[1]

    out_dir = os.path.join(results_dir, "probe")
    os.makedirs(out_dir, exist_ok=True)
    log = setup_logger("run_probe", log_dir=os.path.join(results_dir, "logs"))

    log(f"Linear Probing Experiment")
    log(f"  Model: {cfg['model']['name']} ({model_slug})")
    log(f"  Samples/domain: {spd}")
    log(f"  Components: {components}")
    log(f"  Layers: 0-{num_layers-1}")
    log(f"  Mode: {mode}")
    log()

    # ── Check sample counts ──
    for d in domains:
        n = loader.get_num_samples(d)
        if n < spd:
            log(f"  WARNING: {d} has only {n} samples, using all")

    # ── Run probing ──
    results = {comp: np.zeros(num_layers) for comp in components}
    per_class_results = {comp: {} for comp in components}

    t0 = time.time()
    total_tasks = num_layers * len(components)
    done = 0

    for comp in components:
        for layer_idx in range(num_layers):
            X_parts = []
            y_parts = []

            for label, domain in enumerate(domains):
                n = loader.get_num_samples(domain)
                n_load = min(spd, n)
                sl = slice(0, n_load)

                acts = loader.load_layer(domain, comp, layer_idx, mode, sl)
                X_parts.append(acts)
                y_parts.append(np.full(n_load, label, dtype=np.int32))

            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)

            result = probe_accuracy(X, y, seed=seed)
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
        **{comp: results[comp] for comp in components},
        domains=np.array(domains),
    )

    summary = {
        "model": cfg["model"]["name"],
        "model_slug": model_slug,
        "samples_per_domain": spd,
        "mode": mode,
        "num_layers": num_layers,
        "elapsed_seconds": round(elapsed, 1),
        "components": {},
    }
    for comp in components:
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
    for comp in components:
        accs = results[comp]
        peak = int(np.argmax(accs))
        cv = float(np.std(accs) / np.mean(accs)) if np.mean(accs) > 0 else 0
        log(f"  {comp:<12s} {np.mean(accs):>10.4f} {f'L{peak}':>12s} "
            f"{accs[peak]:>10.4f} {cv:>8.4f}")

    log(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
