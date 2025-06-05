// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
Default-Only Performance Test Shader

This shader implements a 5×5 box filter kernel for testing baseline GPU performance
using default dispatch (no call_group_shape). It serves as a reference implementation
for measuring performance changes across different SlangPy versions.

This is identical to call_group_box_filter.slang but used specifically for
baseline performance measurements.

Key Design Decisions:
- Uses SlangPy's NDBuffer for row-major memory layout testing
- Implements neighborhood sampling (25 reads per thread) to stress memory bandwidth
- Demonstrates coordinate mapping from SlangPy's [z,y,x] call_id ordering
- Uses clamping for boundary conditions to ensure consistent memory access patterns

Memory Access Pattern:
- Each thread reads a 5×5 neighborhood around its assigned pixel
- Default dispatch uses whatever thread group arrangement the GPU driver selects
- This provides a baseline for comparing against specific call group shapes

This shader serves as a baseline workload for measuring GPU memory hierarchy
performance and detecting performance regressions.
*/

import "slangpy";
float4 box_filter_5x5(
    NDBuffer<float4, 2> input_buffer,
    out float4 output_data,
    uint2 grid_cell,
    uint2 call_id
) {
    // call_id passed from Python via spy.call_id()
    // SlangPy uses [z,y,x] ordering, so for 2D call_id is [y,x]:
    // call_id.x = X (width dimension)
    // call_id.y = Y (height dimension)
    uint x = call_id.x;
    uint y = call_id.y;

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
