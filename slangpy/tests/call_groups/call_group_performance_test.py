#!/usr/bin/env python3

"""
Call Group Shape Performance Sample

This sample demonstrates how different call_group_shape configurations affect GPU performance
for memory-intensive compute kernels. It uses a 5×5 box filter as a representative workload
that performs significant memory access (25 reads per output pixel).

Key Features:
- GPU timestamp queries for precise timing measurements (avoids CPU/GPU synchronization overhead)
- Dataset cycling to invalidate GPU cache effects between measurements
- Statistical analysis with warmup periods and outlier exclusion
- Comprehensive comparison of linear vs 2D thread group arrangements

Expected Results:
- Row linear arrangements (ex: 1×32, with [y,x] ordering) typically perform best due to memory coalescing
- Column linear arrangements (ex: 32x1, with [y,x] ordering) perform poorly due to column-major access patterns
- 2D arrangements (4×8, 8×4) offer balanced performance for complex access patterns and have perf similar to row linear.

The test reveals how SlangPy's [z,y,x] coordinate ordering affects memory access patterns
and demonstrates the importance of aligning thread group shapes with data layout.
"""

import slangpy as spy
from slangpy.slangpy import Shape
import numpy as np
import statistics
from typing import Optional

# Note: We are using and expecting [y, x] ordering for everything in this sample
#       so we need to be careful when we call get_call_id<2>() in the shader as
#       slangpy will populate the call_id with [y, x] ordering. Accessing the
#       buffer "image" in column order ([y,x]) is terrible for memory caching /
#       coherency / coalescing. We expect 32x1 (32 rows by 1 column) to be the
#       least performant as every thread will be forced to load a new cache line.
#       1x32 and other row major oriented call group shapes are expected to be
#       more performant, as shifting over by 1 column will likely end up
#       accessing "pixels" in cache lines that have already been loaded.

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

def test_call_group_performance():
    """Test how different call_group_shape configurations affect GPU performance for memory-intensive kernels."""
    print("=== Call Group Shape Performance Test (5×5 Box Filter) ===")

    # Setup
    print("🔧 Setting up device and loading shader module...")
    device = spy.create_device()
    print(f"   ✓ Device created")

    module = spy.Module.load_from_file(device, "call_group_box_filter.slang")
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

    # Test configurations - ordered from worst to best expected performance
    configs = [
        ("32×1 groups", Shape((32, 1))),
        ("8×4 groups", Shape((8, 4))),
        ("4×8 groups", Shape((4, 8))),
        ("2×16 groups", Shape((2, 16))),
        ("1×32 groups", Shape((1, 32))),
        ("No groups (default)", None),
    ]

    results = []

    for config_name, call_group_shape in configs:
        print(f"\n🧪 Testing {config_name}...")

        # Get kernel
        if call_group_shape is not None:
            kernel = module.box_filter_5x5.call_group_shape(call_group_shape)
            print(f"   🔧 Created kernel with call_group_shape: {call_group_shape}")
        else:
            kernel = module.box_filter_5x5
            print(f"   🔧 Using default kernel")

        # Warmup
        print(f"   🔥 Warming up...")
        warmup_data = create_test_data(width, height, seed=42)
        input_buffer.copy_from_numpy(warmup_data)
        for _ in range(warmup_runs):
            kernel(input_buffer, output_buffer, spy.uint2(width, height))
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
                spy.uint2(width, height)
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
        cv = (std_time / avg_time) * 100 if avg_time > 0 else 0

        results.append((config_name, avg_time, std_time, median_time, min_time, cv))
        print(f"   📈 {config_name}: {avg_time:.3f} ± {std_time:.3f} ms (median: {median_time:.3f}, CV: {cv:.1f}%)")

    # Performance analysis
    print(f"\n📊 Performance Analysis:")
    print(f"{'Configuration':<20} {'Mean (ms)':<12} {'Median (ms)':<12} {'Min (ms)':<10} {'CV (%)':<8} {'vs Best':<15}")
    print("-" * 85)

    # Find the best performing configuration
    best_result = min(results, key=lambda x: x[1])
    best_avg = best_result[1]

    for config_name, avg_time, std_time, median_time, min_time, cv in results:
        if config_name == best_result[0]:
            vs_best = "(best)"
        else:
            slowdown = ((avg_time - best_avg) / best_avg) * 100
            vs_best = f"+{slowdown:.1f}% slower"

        print(f"{config_name:<20} {avg_time:<12.2f} {median_time:<12.2f} {min_time:<10.2f} {cv:<8.1f} {vs_best:<15}")

    print(f"\n🏆 Best performing: {best_result[0]} ({best_result[1]:.2f}ms)")
    print(f"Test order: {' → '.join([config[0] for config in configs])}")

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

    return results

def main():
    try:
        results = test_call_group_performance()
        print("\n=== Call Group Shape Performance Test Completed! ===")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
