#!/usr/bin/env python3
"""
finetune_lora.py — Step 3: LoRA fine-tuning with per-epoch checkpointing.

Fine-tunes Qwen2.5-7B with LoRA adapters on a single domain at a time.
Uses a custom PyTorch training loop for full control over gradient
accumulation, mixed-precision, logging, and per-epoch validation/saving.

Usage:
    python finetune_lora.py --domain cs                      # one domain
    python finetune_lora.py --domain cs --config custom.yaml # custom config
    python finetune_lora.py --all-domains                    # all domains sequentially
"""

import argparse
import os
import sys
import time
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_config,
    resolve_path,
    set_seed,
    load_model_and_tokenizer,
    create_dataloader,
    evaluate_perplexity,
    save_json,
    get_device,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning per domain.")
    p.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--domain", type=str, help="Single domain to fine-tune."
    )
    g.add_argument(
        "--all-domains",
        action="store_true",
        help="Fine-tune all domains sequentially.",
    )
    p.add_argument(
        "--resume-from-epoch",
        type=int,
        default=None,
        help="Resume training from this epoch checkpoint (1-indexed).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────
def get_lr_scheduler(optimizer, cfg_training, total_steps):
    """Create LR scheduler based on config."""
    warmup_steps = int(total_steps * cfg_training["warmup_ratio"])
    scheduler_type = cfg_training["lr_scheduler"]

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "constant":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def finetune_domain(domain: str, cfg: dict, script_dir: str):
    """Run full LoRA fine-tuning for one domain."""

    cfg_data = cfg["data"]
    cfg_train = cfg["training"]
    cfg_lora = cfg["lora"]
    cfg_gpu = cfg["gpu"]

    dataset_dir = resolve_path(cfg_data["output_dir"], script_dir)
    adapter_dir = resolve_path(cfg["paths"]["adapter_dir"], script_dir)
    log_dir = resolve_path(cfg["paths"]["log_dir"], script_dir)

    print(f"\n{'=' * 60}")
    print(f"LoRA Fine-Tuning — Domain: {domain}")
    print(f"  Epochs: {cfg_train['num_epochs']}")
    print(f"  Train batch: {cfg_train['train_batch_size']} × "
          f"{cfg_train['gradient_accumulation_steps']} accum = "
          f"{cfg_train['train_batch_size'] * cfg_train['gradient_accumulation_steps']} effective")
    print(f"  Eval batch: {cfg_train['eval_batch_size']}")
    print(f"  LR: {cfg_train['learning_rate']}, Scheduler: {cfg_train['lr_scheduler']}")
    print(f"  LoRA r={cfg_lora['r']}, alpha={cfg_lora['alpha']}")
    print(f"{'=' * 60}")

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, for_training=True)
    device = get_device(cfg)

    # ── Apply LoRA ───────────────────────────────────────────────────────
    task_type_map = {"CAUSAL_LM": TaskType.CAUSAL_LM}
    lora_config = LoraConfig(
        r=cfg_lora["r"],
        lora_alpha=cfg_lora["alpha"],
        lora_dropout=cfg_lora["dropout"],
        target_modules=cfg_lora["target_modules"],
        bias=cfg_lora["bias"],
        task_type=task_type_map[cfg_lora["task_type"]],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── DataParallel (after LoRA applied) ────────────────────────────────
    use_dp = cfg_gpu["multi_gpu"] and torch.cuda.device_count() > 1
    if use_dp:
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Load datasets ────────────────────────────────────────────────────
    train_path = os.path.join(dataset_dir, domain, "train")
    val_path = os.path.join(dataset_dir, domain, "val")

    if not os.path.exists(train_path):
        print(f"  [ERROR] Training data not found: {train_path}")
        return

    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path) if os.path.exists(val_path) else None

    train_texts = train_ds["text"]
    val_texts = val_ds["text"] if val_ds else None

    print(f"  Train: {len(train_texts)} samples")
    if val_texts:
        print(f"  Val:   {len(val_texts)} samples")

    # ── Create DataLoaders ───────────────────────────────────────────────
    train_loader = create_dataloader(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=cfg_data["max_length"],
        batch_size=cfg_train["train_batch_size"],
        shuffle=True,
        num_workers=cfg_train["dataloader_num_workers"],
    )

    val_loader = None
    if val_texts:
        val_loader = create_dataloader(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=cfg_data["max_length"],
            batch_size=cfg_train["eval_batch_size"],
            shuffle=False,
            num_workers=cfg_train["dataloader_num_workers"],
        )

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    # Only optimize LoRA parameters
    base_model = model.module if use_dp else model
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
    )

    num_epochs = cfg_train["num_epochs"]
    steps_per_epoch = math.ceil(
        len(train_loader) / cfg_train["gradient_accumulation_steps"]
    )
    total_steps = steps_per_epoch * num_epochs

    scheduler = get_lr_scheduler(optimizer, cfg_train, total_steps)

    # ── Mixed precision ──────────────────────────────────────────────────
    use_bf16 = cfg_train.get("bf16", False)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # GradScaler is not needed for bf16 (it's loss-scale free)
    scaler = GradScaler(enabled=(not use_bf16))

    # ── Training log ─────────────────────────────────────────────────────
    training_log = {
        "domain": domain,
        "config": {
            "model": cfg["model"]["name"],
            "lora_r": cfg_lora["r"],
            "lora_alpha": cfg_lora["alpha"],
            "lr": cfg_train["learning_rate"],
            "effective_batch_size": (
                cfg_train["train_batch_size"]
                * cfg_train["gradient_accumulation_steps"]
            ),
            "max_length": cfg_data["max_length"],
            "num_epochs": num_epochs,
        },
        "step_logs": [],
        "epoch_logs": [],
        "started_at": datetime.now().isoformat(),
    }

    # ── Training Loop ────────────────────────────────────────────────────
    global_step = 0
    accum_steps = cfg_train["gradient_accumulation_steps"]
    log_interval = cfg_train["logging_steps"]
    max_grad_norm = cfg_train["max_grad_norm"]

    print(f"\n  Total optimizer steps: {total_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Logging every {log_interval} optimizer steps\n")

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'─' * 60}")

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        window_loss = 0.0      # recent window for display
        window_tokens = 0
        micro_step = 0

        pbar = tqdm(
            train_loader,
            desc=f"  [train epoch {epoch}]",
            unit="batch",
            dynamic_ncols=True,
        )

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # DataParallel returns mean loss across GPUs
                if use_dp:
                    loss = loss.mean()

                # Scale for gradient accumulation
                loss_scaled = loss / accum_steps

            # Backward
            if use_bf16:
                loss_scaled.backward()
            else:
                scaler.scale(loss_scaled).backward()

            # Count valid tokens for logging
            n_valid = (labels[:, 1:] != -100).sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_tokens += n_valid
            window_loss += loss.item() * n_valid
            window_tokens += n_valid
            micro_step += 1

            # Optimizer step after accumulation
            if micro_step % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_bf16:
                    nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ──────────────────────────────────────────
                if global_step % log_interval == 0:
                    # Recent windowed loss (shows actual trend)
                    recent_loss = window_loss / max(window_tokens, 1)
                    # Cumulative epoch loss (for JSON log)
                    cumul_loss = epoch_loss / max(epoch_tokens, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    step_log = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": round(recent_loss, 6),
                        "cumulative_loss": round(cumul_loss, 6),
                        "perplexity": round(math.exp(min(recent_loss, 100)), 4),
                        "lr": current_lr,
                    }
                    training_log["step_logs"].append(step_log)

                    pbar.set_postfix({
                        "loss": f"{recent_loss:.4f}",
                        "ppl": f"{math.exp(min(recent_loss, 100)):.1f}",
                        "lr": f"{current_lr:.2e}",
                    })
                    tqdm.write(f"  [Step {global_step}] Loss: {recent_loss:.4f} | PPL: {math.exp(min(recent_loss, 100)):.2f} | LR: {current_lr:.2e}")

                    # Reset window
                    window_loss = 0.0
                    window_tokens = 0

        # ── Epoch summary ────────────────────────────────────────────────
        epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
        epoch_ppl = math.exp(min(epoch_avg_loss, 100))

        print(f"\n  Train — Loss: {epoch_avg_loss:.4f}  |  Perplexity: {epoch_ppl:.2f}")

        # ── Validation ───────────────────────────────────────────────────
        val_metrics = None
        if val_loader:
            print("  Running validation...")
            val_metrics = evaluate_perplexity(model, val_loader, device)
            print(
                f"  Val   — Loss: {val_metrics['loss']:.4f}  |  "
                f"Perplexity: {val_metrics['perplexity']:.2f}"
            )

        # ── Epoch log ────────────────────────────────────────────────────
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(epoch_avg_loss, 6),
            "train_perplexity": round(epoch_ppl, 4),
        }
        if val_metrics:
            epoch_log["val_loss"] = val_metrics["loss"]
            epoch_log["val_perplexity"] = val_metrics["perplexity"]

        training_log["epoch_logs"].append(epoch_log)

        # ── Save adapter ─────────────────────────────────────────────────
        epoch_adapter_dir = os.path.join(adapter_dir, domain, f"epoch_{epoch}")
        os.makedirs(epoch_adapter_dir, exist_ok=True)

        save_model = base_model  # unwrap DP if needed
        save_model.save_pretrained(epoch_adapter_dir)
        print(f"  Adapter saved: {epoch_adapter_dir}")

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    training_log["finished_at"] = datetime.now().isoformat()
    training_log["total_seconds"] = round(elapsed, 2)
    training_log["total_optimizer_steps"] = global_step

    # Save training log
    log_path = os.path.join(log_dir, f"{domain}_training_log.json")
    save_json(training_log, log_path)

    print(f"\n{'=' * 60}")
    print(f"Domain '{domain}' fine-tuning complete!")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final train loss: {epoch_avg_loss:.4f}")
    if val_metrics:
        print(f"  Final val loss:   {val_metrics['loss']:.4f}")
        print(f"  Final val ppl:    {val_metrics['perplexity']:.2f}")
    print(f"  Log: {log_path}")
    print(f"{'=' * 60}\n")

    # Free model from GPU
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = resolve_path(args.config, script_dir)
    cfg = load_config(config_path)

    set_seed(cfg["data"]["seed"])

    if args.all_domains:
        domains = cfg["data"]["domains"]
    else:
        domains = [args.domain]

    for domain in domains:
        finetune_domain(domain, cfg, script_dir)


if __name__ == "__main__":
    main()
