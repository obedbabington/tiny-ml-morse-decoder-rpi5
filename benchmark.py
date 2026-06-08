#!/usr/bin/env python3
"""
Measure TFLite inference performance for the MorseAI model.

Runs repeated inferences against a fixed sample input and prints:
  - model size on disk
  - inference latency (median, mean, p95, min/max)
  - throughput (inferences per second)
  - process memory (RSS) before and after loading the interpreter
  - CPU usage during the benchmark loop

This script does not use GPIO, so you can run it on a Raspberry Pi 5 after
deployment or on a development machine to sanity-check that the model loads.

Usage:
    python3 benchmark.py                 # 10000 iterations, 4 threads
    python3 benchmark.py -n 50000 -t 2   # custom iteration count and thread count

When to run it:
  - After setup, to confirm the model loads and get baseline latency numbers
  - When comparing thread counts (-t 1, 2, or 4) on your board
  - Before updating the performance table in the README with measured values

Memory readings (RSS) require Linux (/proc). On macOS you still get latency
and throughput; run on the Pi for full memory numbers.
"""

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf  # type: ignore

    tflite = tf.lite

MODEL_PATH = Path(__file__).parent / "model" / "morse_classifier.tflite"
CONFIG_PATH = Path(__file__).parent / "model" / "normalization_config.npz"

# Representative "A" input (dot, dash) in microseconds.
SAMPLE_SIGNAL = np.array([150000.0, 350000.0, 0.0, 0.0], dtype=np.float32)


def rss_mb() -> float:
    """Return this process resident set size in MiB (Linux only)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def cpu_times() -> float:
    """Return process CPU time in seconds using nanosecond-resolution clock.

    time.process_time() reads CLOCK_PROCESS_CPUTIME_ID directly, giving
    nanosecond precision. os.times() only has 10 ms (100 Hz jiffy) resolution,
    which rounds to zero for short loops and produces a misleading 0.0% result.
    """
    return time.process_time()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MorseAI TFLite inference latency and resource use."
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=10000,
        help="number of timed inference calls after warmup (default: 10000)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4,
        help="TFLite interpreter thread count (default: 4)",
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=200,
        help="warmup iterations before timing (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MorseAI model benchmark")
    print("=" * 60)

    size_kb = MODEL_PATH.stat().st_size / 1024.0
    print(f"Model file        : {MODEL_PATH.name}")
    print(f"Model size        : {size_kb:.2f} KB ({MODEL_PATH.stat().st_size} bytes)")
    print(f"Threads           : {args.threads}")
    print(f"Iterations        : {args.iterations} (+{args.warmup} warmup)")
    print()

    rss_before = rss_mb()

    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH), num_threads=args.threads)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    rss_after = rss_mb()

    cfg = np.load(str(CONFIG_PATH))
    mean = cfg["mean"].astype(np.float32)
    scale = cfg["scale"].astype(np.float32)
    x = ((SAMPLE_SIGNAL - mean) / scale).reshape(1, 4).astype(np.float32)

    def run_once() -> float:
        t0 = time.perf_counter_ns()
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()
        interpreter.get_tensor(out_idx)
        return (time.perf_counter_ns() - t0) / 1000.0

    for _ in range(args.warmup):
        run_once()

    cpu0 = cpu_times()
    wall0 = time.perf_counter()
    samples = [run_once() for _ in range(args.iterations)]
    wall = time.perf_counter() - wall0
    cpu = cpu_times() - cpu0

    samples.sort()
    p95 = samples[int(0.95 * len(samples)) - 1]
    cpu_pct = (cpu / wall) * 100 if wall > 0 else float("nan")

    print("Results")
    print("-" * 60)
    print(f"Latency median    : {statistics.median(samples):8.1f} us")
    print(f"Latency mean      : {statistics.fmean(samples):8.1f} us")
    print(f"Latency p95       : {p95:8.1f} us")
    print(f"Latency min/max   : {samples[0]:.1f} / {samples[-1]:.1f} us")
    print(f"Throughput        : {args.iterations / wall:8.0f} inferences/s")
    print(
        f"CPU during loop   : {cpu_pct:8.1f} %  "
        "(process CPU time / wall time; ~100% = tight CPU-bound loop)"
    )
    if not np.isnan(rss_after):
        print(f"RSS before load   : {rss_before:8.1f} MiB")
        print(f"RSS after load    : {rss_after:8.1f} MiB")
        print(f"Interpreter cost  : {rss_after - rss_before:8.1f} MiB")
    else:
        print("RSS               : unavailable (run on Linux/Pi for memory numbers)")
    print()
    print("Tip: for peak memory of the full app (GPIO + inference), run:")
    print("     /usr/bin/time -v python3 main.py")


if __name__ == "__main__":
    main()
