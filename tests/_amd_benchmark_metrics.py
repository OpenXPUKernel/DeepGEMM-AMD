"""Optional caller-supplied normalization for private AMD GPU measurements."""

import math


def peak_metrics(total_flops, latency_seconds, world_size, peak_tflops, target_mfu):
    """Never infer a hardware specification from the selected backend."""
    if peak_tflops is None:
        return {}
    if not math.isfinite(peak_tflops) or peak_tflops <= 0:
        raise ValueError("the caller-supplied normalization must be finite and positive")
    if not math.isfinite(target_mfu) or not 0 < target_mfu <= 1:
        raise ValueError("target_mfu must be in (0, 1]")
    if latency_seconds <= 0 or world_size < 1:
        raise ValueError("positive latency and world_size are required")
    denominator = world_size * peak_tflops * 1e12
    mfu = total_flops / (latency_seconds * denominator)
    return {
        "peak_tflops_per_gpu": peak_tflops,
        "mfu": mfu,
        "target_mfu": target_mfu,
        "target_met": mfu >= target_mfu,
        "target_latency_ms": total_flops / (target_mfu * denominator) * 1e3,
        "target_60_percent_ms": total_flops / (0.6 * denominator) * 1e3,
    }
