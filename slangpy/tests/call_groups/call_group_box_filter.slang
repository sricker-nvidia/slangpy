// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
Call Group Shape Performance Test Shader

This shader implements a 5×5 box filter kernel designed to test how different
call_group_shape configurations affect GPU memory access performance.

Key Design Decisions:
- Uses SlangPy's NDBuffer for row-major memory layout testing
- Implements neighborhood sampling (25 reads per thread) to stress memory bandwidth
- Demonstrates coordinate mapping from SlangPy's [z,y,x] call_id ordering
- Uses clamping for boundary conditions to ensure consistent memory access patterns

Memory Access Pattern:
- Each thread reads a 5×5 neighborhood around its assigned pixel
- Linear thread arrangements (1×32) access consecutive memory locations horizontally
- Vertical arrangements (32×1) access memory with large strides, causing poor coalescing
- 2D arrangements (4×8, 8×4) balance spatial locality with memory coalescing

This shader serves as a representative workload for analyzing GPU memory hierarchy
performance and demonstrates the importance of thread group organization.
*/

import "slangpy";
float4 box_filter_5x5(
    NDBuffer<float4, 2> input_buffer,
    out float4 output_data,
    uint2 grid_cell
) {
    int[2] call_id = get_call_id<2>();
    // SlangPy uses [z,y,x] ordering, so for 2D:
    // call_id[0] = Y (height dimension)
    // call_id[1] = X (width dimension)
    uint x = uint(call_id[1]);
    uint y = uint(call_id[0]);

    uint width = uint(input_buffer.shape[1]);
    uint height = uint(input_buffer.shape[0]);

    // Bounds check
    if (x >= width || y >= height)
    {
        output_data = float4(0, 0, 0, 0);
        return float4(0, 0, 0, 0);
    }

    float4 sum = float4(0, 0, 0, 0);

    // 5×5 box filter - access 25 neighboring pixels
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            int sample_x = clamp(int(x) + dx, 0, int(width) - 1);
            int sample_y = clamp(int(y) + dy, 0, int(height) - 1);

            // NDBuffer uses uint2(x, y) indexing
            float4 sample = input_buffer[uint2(sample_x, sample_y)];
            sum += sample;
        }
    }

    // Average the 25 samples
    output_data = sum / 25.0;

    return input_buffer[uint2(x, y)];
}
