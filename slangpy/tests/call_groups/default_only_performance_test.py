#!/usr/bin/env python3

"""
Default-Only Performance Sample (Baseline Measurement)

This sample measures GPU performance for memory-intensive compute kernels using
the default dispatch (no call_group_shape), serving as a baseline for comparison
with other call group configurations.

This is derived from the full call_group_performance_test.py but only runs
the default case to provide baseline performance measurements for comparing
performance changes across different SlangPy versions.

Key Features:
- GPU timestamp queries for precise timing measurements (avoids CPU/GPU synchronization overhead)
- Dataset cycling to invalidate GPU cache effects between measurements
- Statistical analysis with warmup periods and outlier exclusion
- Single configuration testing (default dispatch only)

Use this to measure baseline performance before comparing with different
call group shapes or testing performance regressions.
"""

import slangpy as spy
import numpy as np
import statistics
from typing import Optional

#logger = spy.Logger.get()
#logger.level = spy.LogLevel.debug

def create_test_data(width: int, height: int, seed: Optional[int] = None):
    """Create test image data with optional seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    data = np.random.rand(height, width, 4).astype(np.float32)

    # Add patterns using numpy vectorization
    x_coords = np.arange(width)[np.newaxis, :] / width  # Shape: (1, width)
    y_coords = np.arange(height)[:, np.newaxis] / height  # Shape: (height, 1)

    data[:, :, 0] = x_coords * 0.5 + data[:, :, 0] * 0.5  # X gradient in red channel
    data[:, :, 1] = y_coords * 0.5 + data[:, :, 1] * 0.5  # Y gradient in green channel

    return data

def test_default_performance():
    """Test GPU performance using default dispatch (no call_group_shape) for baseline measurements."""
    print("=== Default-Only Performance Test (5×5 Box Filter) ===")

    # Setup
    print("🔧 Setting up device and loading shader module...")
    device = spy.create_device()
    print(f"   ✓ Device created")

    module = spy.Module.load_from_file(device, "slangpy/tests/call_groups/default_only_box_filter.slang")
    print(f"   ✓ Module loaded")

    # Test parameters
    width, height = 4096, 4096
    total_pixels = width * height
    data_size_mb = (total_pixels * 4 * 4) / (1024 * 1024)
    print(f"Testing {width}×{height} image (5×5 box filter)...")
    print(f"   📊 Total pixels: {total_pixels:,}")
    print(f"   💾 Data size per buffer: {data_size_mb:.1f} MB")

    num_runs = 50
    warmup_runs = 10
    dataset_pool_size = 10
    warmup_count = 5

    # Create GPU buffers
    print("🗄️ Creating GPU buffers...")
    input_buffer = spy.NDBuffer(device, spy.float4, shape=(height, width))
    output_buffer = spy.NDBuffer(device, spy.float4, shape=(height, width))
    print(f"   ✓ Buffers created")

    print(f"\n🧪 Testing Default Dispatch (no call_group_shape)...")

    # Get default kernel
    kernel = module.box_filter_5x5
    print(f"   🔧 Using default kernel")

    # Warmup
    print(f"   🔥 Warming up...")
    warmup_data = create_test_data(width, height, seed=42)
    input_buffer.copy_from_numpy(warmup_data)
    for _ in range(warmup_runs):
        kernel(input_buffer, output_buffer, spy.uint2(width, height), spy.call_id())
    device.wait()
    print(f"   ✓ Warmup completed")

    # Create dataset pool to prevent caching effects
    print(f"   ⏱️ Creating pool of {dataset_pool_size} unique datasets...")
    dataset_pool = []
    for i in range(dataset_pool_size):
        input_data = create_test_data(width, height, seed=i + 1000)
        dataset_pool.append(input_data)

    # GPU timing measurements
    print(f"   📊 Running {num_runs} measurements...")
    queries = device.create_query_pool(spy.QueryType.timestamp, num_runs * 2)

    for i in range(num_runs):
        # Cycle through datasets
        dataset_idx = i % len(dataset_pool)
        input_buffer.copy_from_numpy(dataset_pool[dataset_idx])

        # Progress reporting
        if i == 0:
            print(f"      ✓ Measurement 1: using dataset {dataset_idx}")
        elif i == num_runs - 1:
            print(f"      ✓ Measurement {num_runs}: using dataset {dataset_idx}")
        elif i % 10 == 9:
            print(f"      ✓ Progress: {i+1}/{num_runs}")

        # Time just the kernel execution
        command_encoder = device.create_command_encoder()
        command_encoder.write_timestamp(queries, i * 2)
        kernel.append_to(
            command_encoder,
            input_buffer,
            output_buffer,
            spy.uint2(width, height),
            spy.call_id()
        )
        command_encoder.write_timestamp(queries, i * 2 + 1)
        device.submit_command_buffer(command_encoder.finish())

    device.wait()
    print(f"   ✓ All {num_runs} measurements completed")

    # Process timing results
    gpu_times_sec = np.asarray(queries.get_timestamp_results(0, num_runs * 2))
    all_times = (gpu_times_sec[1::2] - gpu_times_sec[0::2]) * 1000.0

    # Exclude initial measurements to avoid cold start effects
    times = all_times[warmup_count:]
    excluded_times = all_times[:warmup_count]

    print(f"   📈 Retrieved {len(all_times)} GPU timings")
    print(f"   🔥 Excluded first {warmup_count} as warmup: {excluded_times[0]:.1f} to {excluded_times[-1]:.1f}ms")
    print(f"   ⏱️ Using {len(times)} measurements: {times[0]:.1f} to {times[-1]:.1f}ms")

    # Statistical analysis
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    cv = (std_time / avg_time) * 100 if avg_time > 0 else 0

    print(f"\n📊 Default Performance Results:")
    print(f"{'Metric':<15} {'Value':<15}")
    print("-" * 35)
    print(f"{'Mean':<15} {avg_time:.3f} ms")
    print(f"{'Std Dev':<15} {std_time:.3f} ms")
    print(f"{'Median':<15} {median_time:.3f} ms")
    print(f"{'Min':<15} {min_time:.3f} ms")
    print(f"{'Max':<15} {max_time:.3f} ms")
    print(f"{'CV':<15} {cv:.1f}%")

    print(f"\n🏆 Baseline Performance: {avg_time:.3f} ± {std_time:.3f} ms")

    # Additional statistics
    p25 = np.percentile(times, 25)
    p75 = np.percentile(times, 75)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    print(f"\n📈 Percentiles:")
    print(f"   25th: {p25:.3f} ms")
    print(f"   75th: {p75:.3f} ms")
    print(f"   95th: {p95:.3f} ms")
    print(f"   99th: {p99:.3f} ms")

    # Validation
    print(f"\n🔍 Validating results...")
    output_data = output_buffer.to_numpy()
    input_mean = np.mean(warmup_data)
    output_mean = np.mean(output_data)
    input_std = np.std(warmup_data)
    output_std = np.std(output_data)

    print(f"Input mean: {input_mean:.6f}, std: {input_std:.6f}")
    print(f"Output mean: {output_mean:.6f}, std: {output_std:.6f}")
    print(f"Smoothing ratio: {output_std/input_std:.3f}")

    if output_std < input_std:
        print("✓ Box filter working correctly (output smoother than input)")
    else:
        print("⚠ Unexpected results")

    return {
        'mean': avg_time,
        'std': std_time,
        'median': median_time,
        'min': min_time,
        'max': max_time,
        'cv': cv,
        'percentiles': {'p25': p25, 'p75': p75, 'p95': p95, 'p99': p99},
        'all_times': times
    }

def main():
    try:
        results = test_default_performance()
        print("\n=== Default-Only Performance Test Completed! ===")
        print(f"Baseline: {results['mean']:.3f} ± {results['std']:.3f} ms")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
