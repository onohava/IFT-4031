"""Timing and efficiency callbacks for tracking training performance."""
import time
from typing import Optional, Dict, Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TimeToConvergenceCallback(Callback):
    """
    Tracks wall-clock time to convergence during training.

    Logs:
    - elapsed_hours: Total training time so far
    - time_to_best_hours: Time when best loss was achieved
    - best_loss: The best loss value seen
    - best_epoch: Epoch when best loss was achieved
    """

    def __init__(self, monitor: str = "train_loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.start_time: Optional[float] = None
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.time_to_best: float = 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.start_time = time.time()
        self.best_value = None
        self.best_epoch = 0
        self.time_to_best = 0.0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        elapsed_hours = elapsed / 3600

        # Get current monitored value
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is None:
            return

        current_value = current_value.item() if hasattr(current_value, 'item') else float(current_value)

        # Check if this is a new best
        is_best = False
        if self.best_value is None:
            is_best = True
        elif self.mode == "min" and current_value < self.best_value:
            is_best = True
        elif self.mode == "max" and current_value > self.best_value:
            is_best = True

        if is_best:
            self.best_value = current_value
            self.best_epoch = trainer.current_epoch
            self.time_to_best = elapsed_hours

        # Log timing metrics
        pl_module.log("elapsed_hours", elapsed_hours, prog_bar=False)
        pl_module.log("time_to_best_hours", self.time_to_best, prog_bar=False)
        pl_module.log("best_loss", self.best_value, prog_bar=True)
        pl_module.log("best_epoch", float(self.best_epoch), prog_bar=False)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.start_time is None:
            return

        total_time = (time.time() - self.start_time) / 3600

        print("\n" + "=" * 60)
        print("TRAINING TIME SUMMARY")
        print("=" * 60)
        print(f"  Total training time:    {total_time:.2f} hours")
        print(f"  Time to best loss:      {self.time_to_best:.2f} hours")
        print(f"  Best {self.monitor}:          {self.best_value:.6f}")
        print(f"  Best epoch:             {self.best_epoch}")
        print("=" * 60)


class GPUMemoryCallback(Callback):
    """
    Tracks GPU memory usage during training.

    Logs:
    - gpu_memory_allocated_gb: Currently allocated GPU memory
    - gpu_memory_reserved_gb: Total reserved GPU memory
    - gpu_memory_peak_gb: Peak GPU memory usage
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.peak_memory: float = 0.0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        if not torch.cuda.is_available():
            return

        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Get memory stats (in GB)
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9

        self.peak_memory = max(self.peak_memory, peak)

        pl_module.log("gpu_memory_allocated_gb", allocated, prog_bar=False)
        pl_module.log("gpu_memory_reserved_gb", reserved, prog_bar=False)
        pl_module.log("gpu_memory_peak_gb", self.peak_memory, prog_bar=False)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if torch.cuda.is_available():
            print(f"\nPeak GPU Memory: {self.peak_memory:.2f} GB")


class InferenceBenchmarkCallback(Callback):
    """
    Benchmarks inference speed at the end of training.

    Measures:
    - samples_per_second: Generation throughput
    - ms_per_sample: Latency per video sample
    """

    def __init__(
        self,
        num_warmup: int = 2,
        num_benchmark: int = 8,
        batch_size: int = 4,
    ):
        super().__init__()
        self.num_warmup = num_warmup
        self.num_benchmark = num_benchmark
        self.batch_size = batch_size
        self.results: Optional[Dict[str, float]] = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not hasattr(pl_module, 'sample_videos'):
            print("Warning: Model does not have sample_videos method, skipping benchmark")
            return

        device = next(pl_module.parameters()).device
        pl_module.eval()

        print("\n" + "=" * 60)
        print("INFERENCE BENCHMARK")
        print("=" * 60)
        print(f"  Warming up ({self.num_warmup} runs)...")

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = pl_module.sample_videos(num_samples=self.batch_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        print(f"  Benchmarking ({self.num_benchmark} runs, batch_size={self.batch_size})...")
        times = []

        with torch.no_grad():
            for _ in range(self.num_benchmark):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = pl_module.sample_videos(num_samples=self.batch_size)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        samples_per_second = self.batch_size / avg_time
        ms_per_sample = (avg_time / self.batch_size) * 1000

        self.results = {
            "samples_per_second": samples_per_second,
            "ms_per_sample": ms_per_sample,
            "avg_batch_time_seconds": avg_time,
            "std_batch_time_seconds": std_time,
        }

        print(f"\n  Results:")
        print(f"    Throughput:     {samples_per_second:.2f} samples/sec")
        print(f"    Latency:        {ms_per_sample:.1f} ms/sample")
        print(f"    Batch time:     {avg_time:.2f} ± {std_time:.2f} sec")
        print("=" * 60)

        # Log to wandb if available
        if trainer.logger:
            trainer.logger.log_metrics({
                "inference/samples_per_second": samples_per_second,
                "inference/ms_per_sample": ms_per_sample,
            })


def benchmark_inference(
    model,
    num_samples: int = 4,
    num_warmup: int = 2,
    num_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Standalone function to benchmark inference speed.

    Args:
        model: VideoDiffusionModel or DiffusionLightningModule
        num_samples: Batch size for generation
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        device: Device to run on

    Returns:
        Dictionary with timing results
    """
    model = model.to(device)
    model.eval()

    # Get the sample method
    if hasattr(model, 'sample_videos'):
        sample_fn = lambda: model.sample_videos(num_samples=num_samples)
    elif hasattr(model, 'sample'):
        sample_fn = lambda: model.sample(batch_size=num_samples)
    else:
        raise ValueError("Model must have sample_videos or sample method")

    print(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = sample_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print(f"Benchmarking ({num_runs} runs)...")
    times = []

    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = sample_fn()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    samples_per_second = num_samples / avg_time
    ms_per_sample = (avg_time / num_samples) * 1000

    results = {
        "samples_per_second": samples_per_second,
        "ms_per_sample": ms_per_sample,
        "avg_batch_time_seconds": avg_time,
        "std_batch_time_seconds": std_time,
        "batch_size": num_samples,
        "num_runs": num_runs,
    }

    if torch.cuda.is_available():
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nResults:")
    print(f"  Throughput:     {samples_per_second:.2f} samples/sec")
    print(f"  Latency:        {ms_per_sample:.1f} ms/sample")
    print(f"  Batch time:     {avg_time:.2f} ± {std_time:.2f} sec")
    if "peak_memory_gb" in results:
        print(f"  Peak memory:    {results['peak_memory_gb']:.2f} GB")

    return results
