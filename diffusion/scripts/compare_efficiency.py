#!/usr/bin/env python
"""
Compare efficiency metrics between VAE and KAE diffusion training runs.
Extracts time_to_best_hours, best_loss, and other metrics from WandB or logs.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import subprocess


def get_wandb_metrics(run_name: str, project: str = "koopman-diffusion") -> Optional[Dict]:
    """Fetch metrics from WandB run."""
    try:
        import wandb
        api = wandb.Api()

        # Find run by name
        runs = api.runs(f"{project}", filters={"display_name": run_name})
        for run in runs:
            summary = run.summary._json_dict
            return {
                "run_name": run.name,
                "best_loss": summary.get("best_loss"),
                "time_to_best_hours": summary.get("time_to_best_hours"),
                "elapsed_hours": summary.get("elapsed_hours"),
                "best_epoch": summary.get("best_epoch"),
                "gpu_memory_peak_gb": summary.get("gpu_memory_peak_gb"),
                "samples_per_second": summary.get("inference/samples_per_second"),
                "ms_per_sample": summary.get("inference/ms_per_sample"),
            }
    except Exception as e:
        print(f"Could not fetch WandB metrics: {e}")
    return None


def parse_log_file(log_path: str) -> Dict:
    """Parse training log file for efficiency metrics."""
    metrics = {}

    with open(log_path) as f:
        content = f.read()

    # Parse training time summary
    if "TRAINING TIME SUMMARY" in content:
        lines = content.split("TRAINING TIME SUMMARY")[1].split("="*60)[0]
        for line in lines.split("\n"):
            if "Total training time:" in line:
                metrics["elapsed_hours"] = float(line.split(":")[1].strip().split()[0])
            elif "Time to best loss:" in line:
                metrics["time_to_best_hours"] = float(line.split(":")[1].strip().split()[0])
            elif "Best train_loss:" in line:
                metrics["best_loss"] = float(line.split(":")[1].strip())
            elif "Best epoch:" in line:
                metrics["best_epoch"] = int(line.split(":")[1].strip())

    # Parse GPU memory
    if "Peak GPU Memory:" in content:
        for line in content.split("\n"):
            if "Peak GPU Memory:" in line:
                metrics["gpu_memory_peak_gb"] = float(line.split(":")[1].strip().split()[0])

    # Parse inference benchmark
    if "INFERENCE BENCHMARK" in content:
        lines = content.split("INFERENCE BENCHMARK")[1].split("="*60)[0]
        for line in lines.split("\n"):
            if "Throughput:" in line:
                metrics["samples_per_second"] = float(line.split(":")[1].strip().split()[0])
            elif "Latency:" in line:
                metrics["ms_per_sample"] = float(line.split(":")[1].strip().split()[0])

    return metrics


def format_table(vae_metrics: Dict, kae_metrics: Dict) -> str:
    """Format comparison as LaTeX table."""

    def fmt(val, precision=2):
        if val is None:
            return "--"
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)

    table = r"""
\begin{table}[h]
\centering
\caption{Diffusion Training Efficiency Comparison (Moving MNIST)}
\label{tab:efficiency}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Loss} $\downarrow$ & \textbf{Time to Best} & \textbf{Best Epoch} & \textbf{Peak Memory} & \textbf{Inference} \\
               &                            & (hours)               &                     & (GB)                 & (ms/sample)        \\
\midrule
"""

    # VAE row
    table += f"Diff + VAE & {fmt(vae_metrics.get('best_loss'), 4)} & "
    table += f"{fmt(vae_metrics.get('time_to_best_hours'))} & "
    table += f"{fmt(vae_metrics.get('best_epoch'), 0)} & "
    table += f"{fmt(vae_metrics.get('gpu_memory_peak_gb'))} & "
    table += f"{fmt(vae_metrics.get('ms_per_sample'), 1)} \\\\\n"

    # KAE row (bold if better)
    table += r"\textbf{Diff + KAE} & "
    table += f"\\textbf{{{fmt(kae_metrics.get('best_loss'), 4)}}} & "
    table += f"{fmt(kae_metrics.get('time_to_best_hours'))} & "
    table += f"{fmt(kae_metrics.get('best_epoch'), 0)} & "
    table += f"{fmt(kae_metrics.get('gpu_memory_peak_gb'))} & "
    table += f"{fmt(kae_metrics.get('ms_per_sample'), 1)} \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table


def main():
    parser = argparse.ArgumentParser(description="Compare VAE vs KAE efficiency")
    parser.add_argument("--vae-log", type=str, help="Path to VAE diffusion log file")
    parser.add_argument("--kae-log", type=str, help="Path to KAE diffusion log file")
    parser.add_argument("--vae-run", type=str, default="diffusion-vae-compare",
                        help="WandB run name for VAE")
    parser.add_argument("--kae-run", type=str, default="diffusion-kae-compare",
                        help="WandB run name for KAE")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    # Get metrics from WandB or logs
    vae_metrics = {}
    kae_metrics = {}

    # Try WandB first
    print("Fetching metrics from WandB...")
    vae_wandb = get_wandb_metrics(args.vae_run)
    kae_wandb = get_wandb_metrics(args.kae_run)

    if vae_wandb:
        vae_metrics.update(vae_wandb)
        print(f"  VAE metrics from WandB: {vae_wandb}")
    if kae_wandb:
        kae_metrics.update(kae_wandb)
        print(f"  KAE metrics from WandB: {kae_wandb}")

    # Supplement with log files if provided
    if args.vae_log and Path(args.vae_log).exists():
        log_metrics = parse_log_file(args.vae_log)
        vae_metrics.update({k: v for k, v in log_metrics.items() if k not in vae_metrics})
        print(f"  VAE metrics from log: {log_metrics}")

    if args.kae_log and Path(args.kae_log).exists():
        log_metrics = parse_log_file(args.kae_log)
        kae_metrics.update({k: v for k, v in log_metrics.items() if k not in kae_metrics})
        print(f"  KAE metrics from log: {log_metrics}")

    # Print comparison
    print("\n" + "=" * 70)
    print("EFFICIENCY COMPARISON: VAE vs KAE")
    print("=" * 70)

    metrics_to_compare = [
        ("best_loss", "Best Loss", "lower"),
        ("time_to_best_hours", "Time to Best (hours)", "lower"),
        ("best_epoch", "Best Epoch", "info"),
        ("gpu_memory_peak_gb", "Peak GPU Memory (GB)", "lower"),
        ("ms_per_sample", "Inference Latency (ms)", "lower"),
        ("samples_per_second", "Throughput (samples/s)", "higher"),
    ]

    for key, label, direction in metrics_to_compare:
        vae_val = vae_metrics.get(key)
        kae_val = kae_metrics.get(key)

        vae_str = f"{vae_val:.4f}" if isinstance(vae_val, float) else str(vae_val or "--")
        kae_str = f"{kae_val:.4f}" if isinstance(kae_val, float) else str(kae_val or "--")

        winner = ""
        if vae_val is not None and kae_val is not None and direction != "info":
            if direction == "lower":
                winner = " <-- better" if kae_val < vae_val else ""
            else:
                winner = " <-- better" if kae_val > vae_val else ""

        print(f"{label:30s}  VAE: {vae_str:12s}  KAE: {kae_str:12s}{winner}")

    print("=" * 70)

    # Generate LaTeX table
    print("\nLaTeX Table:")
    print(format_table(vae_metrics, kae_metrics))

    # Save results
    if args.output:
        results = {
            "vae": vae_metrics,
            "kae": kae_metrics,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
