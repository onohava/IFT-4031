#!/usr/bin/env python
"""
Compare efficiency metrics between VAE and KAE diffusion training on UCF101.
Extracts time_to_best_hours, best_loss, GPU memory, and inference metrics.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


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


def format_markdown_table(vae_metrics: Dict, kae_metrics: Dict) -> str:
    """Format comparison as Markdown table."""

    def fmt(val, precision=4):
        if val is None:
            return "--"
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)

    def winner_mark(vae_val, kae_val, direction):
        if vae_val is None or kae_val is None:
            return "", ""
        if direction == "lower":
            if kae_val < vae_val:
                return "", " **"
            elif vae_val < kae_val:
                return " **", ""
        else:  # higher is better
            if kae_val > vae_val:
                return "", " **"
            elif vae_val > kae_val:
                return " **", ""
        return "", ""

    table = """
## UCF101 ApplyLipstick - Diffusion Training Efficiency Comparison

| Metric | Diffusion + VAE | Diffusion + KAE | Winner |
|--------|-----------------|-----------------|--------|
"""

    metrics_config = [
        ("best_loss", "Best Loss", "lower", 4),
        ("time_to_best_hours", "Time to Best (hours)", "lower", 2),
        ("elapsed_hours", "Total Time (hours)", "lower", 2),
        ("best_epoch", "Best Epoch", "info", 0),
        ("gpu_memory_peak_gb", "Peak GPU Memory (GB)", "lower", 2),
        ("ms_per_sample", "Inference Latency (ms)", "lower", 1),
        ("samples_per_second", "Throughput (samples/s)", "higher", 2),
    ]

    for key, label, direction, precision in metrics_config:
        vae_val = vae_metrics.get(key)
        kae_val = kae_metrics.get(key)

        vae_str = fmt(vae_val, precision)
        kae_str = fmt(kae_val, precision)

        if direction == "info":
            winner = "--"
        elif vae_val is None or kae_val is None:
            winner = "--"
        elif direction == "lower":
            winner = "KAE" if kae_val < vae_val else ("VAE" if vae_val < kae_val else "Tie")
        else:
            winner = "KAE" if kae_val > vae_val else ("VAE" if vae_val > kae_val else "Tie")

        table += f"| {label} | {vae_str} | {kae_str} | {winner} |\n"

    return table


def format_latex_table(vae_metrics: Dict, kae_metrics: Dict) -> str:
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
\caption{Diffusion Training Efficiency on UCF101 (ApplyLipstick)}
\label{tab:ucf101-efficiency}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Loss} $\downarrow$ & \textbf{Time to Best} & \textbf{Peak Memory} & \textbf{Inference} \\
               &                            & (hours)               & (GB)                 & (ms/sample)        \\
\midrule
"""

    # VAE row
    table += f"Diff + VAE & {fmt(vae_metrics.get('best_loss'), 4)} & "
    table += f"{fmt(vae_metrics.get('time_to_best_hours'))} & "
    table += f"{fmt(vae_metrics.get('gpu_memory_peak_gb'))} & "
    table += f"{fmt(vae_metrics.get('ms_per_sample'), 1)} \\\\\n"

    # KAE row
    table += f"Diff + KAE & {fmt(kae_metrics.get('best_loss'), 4)} & "
    table += f"{fmt(kae_metrics.get('time_to_best_hours'))} & "
    table += f"{fmt(kae_metrics.get('gpu_memory_peak_gb'))} & "
    table += f"{fmt(kae_metrics.get('ms_per_sample'), 1)} \\\\\n"

    table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table


def main():
    parser = argparse.ArgumentParser(description="Compare VAE vs KAE efficiency on UCF101")
    parser.add_argument("--vae-log", type=str,
                        default="/private/home/soniajoseph/IFT-4031/diffusion/outputs/logs/diff_vae_4gpu_latest.out",
                        help="Path to VAE diffusion log file")
    parser.add_argument("--kae-log", type=str,
                        default="/private/home/soniajoseph/IFT-4031/diffusion/outputs/logs/diff_kae_4gpu_latest.out",
                        help="Path to KAE diffusion log file")
    parser.add_argument("--vae-run", type=str, default="diffusion-vae-ucf101-4gpu",
                        help="WandB run name for VAE")
    parser.add_argument("--kae-run", type=str, default="diffusion-ucf101-4gpu",
                        help="WandB run name for KAE")
    parser.add_argument("--output", type=str,
                        default="/private/home/soniajoseph/IFT-4031/diffusion/outputs/ucf101_efficiency_report.json",
                        help="Output JSON file")
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
        print(f"  VAE metrics from WandB: OK")
    if kae_wandb:
        kae_metrics.update(kae_wandb)
        print(f"  KAE metrics from WandB: OK")

    # Supplement with log files if provided
    if args.vae_log and Path(args.vae_log).exists():
        log_metrics = parse_log_file(args.vae_log)
        vae_metrics.update({k: v for k, v in log_metrics.items() if k not in vae_metrics or vae_metrics[k] is None})
        print(f"  VAE metrics from log: OK")

    if args.kae_log and Path(args.kae_log).exists():
        log_metrics = parse_log_file(args.kae_log)
        kae_metrics.update({k: v for k, v in log_metrics.items() if k not in kae_metrics or kae_metrics[k] is None})
        print(f"  KAE metrics from log: OK")

    # Print comparison
    print("\n" + "=" * 80)
    print("UCF101 EFFICIENCY COMPARISON: Diffusion + VAE vs Diffusion + KAE")
    print("=" * 80)

    metrics_to_compare = [
        ("best_loss", "Best Loss", "lower"),
        ("time_to_best_hours", "Time to Best (hours)", "lower"),
        ("elapsed_hours", "Total Training Time (hours)", "lower"),
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
                winner = " <-- KAE better" if kae_val < vae_val else (" <-- VAE better" if vae_val < kae_val else "")
            else:
                winner = " <-- KAE better" if kae_val > vae_val else (" <-- VAE better" if vae_val > kae_val else "")

        print(f"{label:35s}  VAE: {vae_str:12s}  KAE: {kae_str:12s}{winner}")

    print("=" * 80)

    # Generate tables
    print("\n" + format_markdown_table(vae_metrics, kae_metrics))
    print("\nLaTeX Table:")
    print(format_latex_table(vae_metrics, kae_metrics))

    # Save results
    if args.output:
        results = {
            "dataset": "UCF101-ApplyLipstick",
            "vae": vae_metrics,
            "kae": kae_metrics,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
