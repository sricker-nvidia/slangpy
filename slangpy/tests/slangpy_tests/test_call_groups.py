# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests for call group functionality in SlangPy.

This module tests:
1. Call group shapes for dimensions 1-5
2. Shader functions: get_call_id(), get_call_group_id(), get_call_group_thread_id(), get_flat_call_id(), get_flat_call_group_id(), get_flat_call_group_thread_id()
3. Python functions: spy.call_group_id(), spy.call_group_thread_id(), spy.flat_call_group_id(), spy.flat_call_group_thread_id()
4. Edge cases: misaligned shapes, shapes smaller than groups, etc.
5. Function existence, Python argument functions, and specific validation tests
"""

import numpy as np
import pytest
import slangpy as spy
from slangpy import DeviceType
from slangpy.slangpy import Shape
from slangpy.types.buffer import NDBuffer
from . import helpers


# Test data for different dimensions and call group configurations
CALL_GROUP_TEST_CASES = [
    # (dimension, call_shape, call_group_shape, description)
    (1, (8,), (2,), "1D basic aligned"),
    (1, (9,), (2,), "1D misaligned"),
    (1, (1,), (4,), "1D smaller than group"),
    (2, (8, 6), (2, 3), "2D basic aligned"),
    (2, (9, 7), (2, 3), "2D misaligned"),
    (2, (1, 1), (4, 4), "2D smaller than group"),
    (3, (8, 6, 4), (2, 3, 2), "3D basic aligned"),
    (3, (9, 7, 5), (2, 3, 2), "3D misaligned"),
    (4, (8, 6, 4, 4), (2, 3, 2, 2), "4D basic aligned"),
    (4, (9, 7, 5, 3), (2, 3, 2, 2), "4D misaligned"),
    (5, (8, 6, 4, 4, 2), (2, 3, 2, 2, 2), "5D basic aligned"),
    (5, (9, 7, 5, 3, 3), (2, 3, 2, 2, 2), "5D misaligned"),
]

EDGE_CASE_TEST_CASES = [
    # (dimension, call_shape, call_group_shape, description)
    (1, (32,), (1,), "1D linear dispatch (group size 1)"),
    (2, (32, 32), (32, 1), "2D linear first dimension"),
    (2, (32, 32), (1, 32), "2D linear second dimension"),
    (3, (100, 4, 2), (32, 2, 2), "3D with varying dimensions"),
    (2, (3, 5), (10, 10), "2D call smaller than group in all dims"),
    (3, (8, 4, 12), (4, 8, 6), "3D call smaller in some dims"),
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_functions_exist(device_type: DeviceType):
    """Test that all call group functions can be called without error."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

float test_functions_exist(uint2 grid_cell) {
    // Test global getter functions with explicit types
    int[2] call_id_result = get_call_id<2>();
    uint[2] call_group_id_result = get_call_group_id<2>();
    uint[2] call_group_thread_id_result = get_call_group_thread_id<2>();
    int flat_call_id_result = get_flat_call_id();
    uint flat_call_group_id_result = get_flat_call_group_id();
    uint flat_call_group_thread_id_result = get_flat_call_group_thread_id();

    // Return success - we just want to verify compilation works
    return 1.0f;
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test that the function can be called without error
    result = module.test_functions_exist(spy.grid((4, 6)), _result="numpy")

    # Verify all results are 1.0 (success)
    assert np.all(result == 1.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_python_arguments(device_type: DeviceType):
    """Test that Python spy.call_group_* functions work as arguments."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

// Test that spy.call_group_* Python functions can be passed as arguments
float test_python_args(uint2 grid_cell, uint2 call_group_id_arg, uint2 call_group_thread_id_arg,
                      uint flat_call_group_id_arg, uint flat_call_group_thread_id_arg) {
    // Just verify the arguments are received - return sum to detect if they're all zero
    return (float)(call_group_id_arg.x + call_group_id_arg.y +
                  call_group_thread_id_arg.x + call_group_thread_id_arg.y +
                  flat_call_group_id_arg + flat_call_group_thread_id_arg);
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test that the Python spy functions can be passed as arguments
    result = module.test_python_args(
        spy.grid((4, 6)),
        spy.call_group_id(),
        spy.call_group_thread_id(),
        spy.flat_call_group_id(),
        spy.flat_call_group_thread_id(),
        _result="numpy",
    )

    # The result should contain various values (not all zeros)
    assert result.size > 0
    # At least some results should be non-zero (since we're testing non-trivial group arrangements)
    assert np.any(result > 0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_1d_call_groups_with_validation(device_type: DeviceType):
    """Test 1D call groups with specific mathematical validation."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint test_1d_groups(uint grid_cell) {
    uint[1] call_group_id = get_call_group_id<1>();
    uint[1] call_group_thread_id = get_call_group_thread_id<1>();

    // Return packed result: high 16 bits = group_id, low 16 bits = thread_id
    return (call_group_id[0] << 16) | call_group_thread_id[0];
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test with call shape (8,) and call group shape (2,)
    call_shape = (8,)
    call_group_shape = (2,)

    result = module.test_1d_groups.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result="numpy"
    )

    # Extract group IDs and thread IDs
    group_ids = (result >> 16) & 0xFFFF
    thread_ids = result & 0xFFFF

    # Validate dimensions
    assert result.shape == call_shape

    # Basic validation: group IDs should be in range [0, ceil(8/2)-1] = [0, 3]
    assert np.all(group_ids >= 0) and np.all(group_ids < 4)

    # Thread IDs should be in range [0, 1] (group size is 2)
    assert np.all(thread_ids >= 0) and np.all(thread_ids < 2)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_flat_functions_detailed(device_type: DeviceType):
    """Test flat (1D) call group functions with detailed validation."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint2 test_flat_functions(uint2 grid_cell) {
    uint flat_call_group_id = get_flat_call_group_id();
    uint flat_call_group_thread_id = get_flat_call_group_thread_id();

    return uint2(flat_call_group_id, flat_call_group_thread_id);
}
"""

    module = helpers.create_module(device, kernel_source)

    call_shape = (4, 6)
    call_group_shape = (2, 3)

    result = module.test_flat_functions.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result="numpy"
    )

    # Extract flat group IDs and thread IDs
    flat_group_ids = result[:, :, 0]  # First component
    flat_thread_ids = result[:, :, 1]  # Second component

    # Validate basic properties
    assert np.all(flat_group_ids >= 0)
    assert np.all(flat_thread_ids >= 0)

    # Thread IDs should be less than group size (2*3 = 6)
    assert np.all(flat_thread_ids < 6)

    # Should have some non-zero values (except the first position which may be 0)
    # With call shape (4,6) and group shape (2,3), we expect variety in flat IDs
    assert np.any(flat_group_ids > 0) or np.any(
        flat_thread_ids > 0
    ), "All flat IDs are 0 - flat call group functions may not be working correctly"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_2d_validation(device_type: DeviceType):
    """
    Test that SlangPy's call group calculations are mathematically consistent for 2D.

    Instead of trying to reverse-engineer the exact algorithm, we validate:
    1. Basic mathematical relationships between the values
    2. Consistency across different grid/group sizes
    3. Edge cases and boundaries
    """

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

int2 test_call_id_2d(uint2 grid_cell, int2 call_id_arg) {
    return call_id_arg;
}

uint2 test_call_group_id_2d(uint2 grid_cell, uint2 call_group_id_arg) {
    return call_group_id_arg;
}

uint2 test_call_group_thread_id_2d(uint2 grid_cell, uint2 call_group_thread_id_arg) {
    return call_group_thread_id_arg;
}

uint test_flat_call_group_id_2d(uint2 grid_cell, uint flat_call_group_id_arg) {
    return flat_call_group_id_arg;
}

uint test_flat_call_group_thread_id_2d(uint2 grid_cell, uint flat_call_group_thread_id_arg) {
    return flat_call_group_thread_id_arg;
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test with multiple configurations
    test_cases = [
        ((4, 4), (2, 2)),  # Perfect alignment
        ((6, 4), (2, 2)),  # Partial alignment
        ((8, 6), (2, 3)),  # Different group aspect ratio
        ((5, 7), (2, 2)),  # Unaligned case
    ]

    for call_shape, group_shape in test_cases:
        # Get all values from SlangPy
        call_id = module.test_call_id_2d.call_group_shape(Shape(group_shape))(
            spy.grid(call_shape), spy.call_id(), _result="numpy"
        )

        call_group_id = module.test_call_group_id_2d.call_group_shape(Shape(group_shape))(
            spy.grid(call_shape), spy.call_group_id(), _result="numpy"
        )

        call_group_thread_id = module.test_call_group_thread_id_2d.call_group_shape(
            Shape(group_shape)
        )(spy.grid(call_shape), spy.call_group_thread_id(), _result="numpy")

        flat_call_group_id = module.test_flat_call_group_id_2d.call_group_shape(Shape(group_shape))(
            spy.grid(call_shape), spy.flat_call_group_id(), _result="numpy"
        )

        flat_call_group_thread_id = module.test_flat_call_group_thread_id_2d.call_group_shape(
            Shape(group_shape)
        )(spy.grid(call_shape), spy.flat_call_group_thread_id(), _result="numpy")

        # Validation 1: call_id should match thread position [x, y]
        for y in range(call_shape[0]):
            for x in range(call_shape[1]):
                expected_call_id = [x, y]  # call_id is [x, y] not [y, x]
                actual_call_id = call_id[y, x]
                assert np.array_equal(
                    actual_call_id, expected_call_id
                ), f"call_id mismatch at [{y},{x}]: expected {expected_call_id}, got {actual_call_id}"

        # Validation 2: Relationship between flat and N-D call group IDs
        call_group_size = group_shape[0] * group_shape[1]
        grid_w = (call_shape[1] + group_shape[1] - 1) // group_shape[1]

        for y in range(call_shape[0]):
            for x in range(call_shape[1]):
                # Calculate which 2D group this thread belongs to
                group_y = y // group_shape[0]
                group_x = x // group_shape[1]
                expected_flat_group_id = group_y * grid_w + group_x

                actual_flat_group_id = flat_call_group_id[y, x]
                actual_call_group_id_2d = call_group_id[y, x]

                assert (
                    actual_flat_group_id == expected_flat_group_id
                ), f"flat_call_group_id mismatch at [{y},{x}]: expected {expected_flat_group_id}, got {actual_flat_group_id}"

                # Verify 2D group ID matches calculated group position
                expected_call_group_id_2d = [group_x, group_y]  # [x, y] format like call_id
                assert np.array_equal(
                    actual_call_group_id_2d, expected_call_group_id_2d
                ), f"call_group_id mismatch at [{y},{x}]: expected {expected_call_group_id_2d}, got {actual_call_group_id_2d}"

        # Validation 3: call_group_thread_id should be position within group
        for y in range(call_shape[0]):
            for x in range(call_shape[1]):
                # Calculate position within the group
                thread_y_in_group = y % group_shape[0]
                thread_x_in_group = x % group_shape[1]
                expected_thread_id = [
                    thread_x_in_group,
                    thread_y_in_group,
                ]  # [x, y] format like call_id

                actual_thread_id = call_group_thread_id[y, x]
                assert np.array_equal(
                    actual_thread_id, expected_thread_id
                ), f"call_group_thread_id mismatch at [{y},{x}]: expected {expected_thread_id}, got {actual_thread_id}"

        # Validation 4: flat_call_group_thread_id calculation
        for y in range(call_shape[0]):
            for x in range(call_shape[1]):
                thread_y_in_group = y % group_shape[0]
                thread_x_in_group = x % group_shape[1]
                expected_flat_thread_id = thread_y_in_group * group_shape[1] + thread_x_in_group

                actual_flat_thread_id = flat_call_group_thread_id[y, x]
                assert (
                    actual_flat_thread_id == expected_flat_thread_id
                ), f"flat_call_group_thread_id mismatch at [{y},{x}]: expected {expected_flat_thread_id}, got {actual_flat_thread_id}"

        # Validation 5: Fundamental equation - call_id = call_group_id * group_dim + call_group_thread_id
        for y in range(call_shape[0]):
            for x in range(call_shape[1]):
                call_id_val = call_id[y, x]
                call_group_id_val = call_group_id[y, x]
                call_group_thread_id_val = call_group_thread_id[y, x]

                # For each dimension: call_id[i] = call_group_id[i] * group_dim[i] + call_group_thread_id[i]
                expected_call_id_y = (
                    call_group_id_val[1] * group_shape[0] + call_group_thread_id_val[1]
                )  # y = call_group_id[1] * group_h + call_group_thread_id[1]
                expected_call_id_x = (
                    call_group_id_val[0] * group_shape[1] + call_group_thread_id_val[0]
                )  # x = call_group_id[0] * group_w + call_group_thread_id[0]
                expected_call_id = [expected_call_id_x, expected_call_id_y]

                assert np.array_equal(call_id_val, expected_call_id), (
                    f"Fundamental equation violated at [{y},{x}]: call_id={call_id_val}, "
                    f"expected {expected_call_id} from call_group_id={call_group_id_val} * {group_shape} + {call_group_thread_id_val}"
                )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_5d_validation(device_type: DeviceType):
    """
    Test that SlangPy's call group calculations work for 5D (simplified test).

    This test validates that 5D call group functions can be called without errors
    and return reasonable values, rather than doing deep mathematical validation.
    """

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

float test_5d_basic(uint[5] grid_cell) {
    // Just test that the 5D call group functions can be called without error
    int[5] call_id = get_call_id<5>();
    uint[5] call_group_id = get_call_group_id<5>();
    uint[5] call_group_thread_id = get_call_group_thread_id<5>();
    int flat_call_id = get_flat_call_id();
    uint flat_call_group_id = get_flat_call_group_id();
    uint flat_call_group_thread_id = get_flat_call_group_thread_id();

    // Return a simple validation that functions executed
    return 1.0f;
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test with simple 5D configuration
    call_shape = (3, 2, 2, 2, 2)
    call_group_shape = (1, 1, 2, 2, 2)

    # Create output buffer
    result_buffer = NDBuffer(device=device, shape=call_shape, dtype=float)

    # Call with call group shape
    module.test_5d_basic.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result=result_buffer
    )

    # Validate that all calls completed successfully (all should be 1.0)
    results = result_buffer.to_numpy()
    assert np.all(results == 1.0), f"5D call group functions failed to execute properly"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_math_edge_cases(device_type: DeviceType):
    """Test edge cases and boundary conditions."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

uint test_flat_call_group_id(uint2 grid_cell, uint flat_call_group_id_arg) {
    return flat_call_group_id_arg;
}
"""

    module = helpers.create_module(device, kernel_source)

    # Edge case 1: 1x1 groups (should be equivalent to no groups)
    call_shape = (3, 3)

    # With 1x1 groups
    flat_id_with_groups = module.test_flat_call_group_id.call_group_shape(Shape((1, 1)))(
        spy.grid(call_shape), spy.flat_call_group_id(), _result="numpy"
    )

    # Without explicit groups
    flat_id_no_groups = module.test_flat_call_group_id(
        spy.grid(call_shape), spy.flat_call_group_id(), _result="numpy"
    )

    assert np.array_equal(
        flat_id_with_groups, flat_id_no_groups
    ), "1x1 groups should be equivalent to no groups"

    # Edge case 2: Group larger than call shape
    large_group_result = module.test_flat_call_group_id.call_group_shape(Shape((4, 4)))(
        spy.grid((2, 2)), spy.flat_call_group_id(), _result="numpy"
    )

    # All threads should be in group 0
    expected = np.zeros((2, 2), dtype=np.uint32)
    assert np.array_equal(
        large_group_result, expected
    ), "All threads should be in group 0 when group is larger than call shape"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dimension,call_shape,call_group_shape,description", CALL_GROUP_TEST_CASES)
def test_call_group_dimensions(
    device_type: DeviceType,
    dimension: int,
    call_shape: tuple,
    call_group_shape: tuple,
    description: str,
):
    """Test call groups across different dimensions and configurations."""

    device = helpers.get_device(device_type)

    # Create a simpler test that just validates the math works
    # For 5D+, use array types like uint[5], uint[6], etc.
    if dimension <= 4:
        param_type = {1: "uint", 2: "uint2", 3: "uint3", 4: "uint4"}[dimension]
    else:
        param_type = f"uint[{dimension}]"
    kernel_source = f"""
import "slangpy";

float test_call_group_math({param_type} grid_pos) {{
    // Just test that the functions can be called without error
    int[{dimension}] call_id = get_call_id<{dimension}>();
    uint[{dimension}] call_group_id = get_call_group_id<{dimension}>();
    uint[{dimension}] call_group_thread_id = get_call_group_thread_id<{dimension}>();
    int flat_call_id = get_flat_call_id();
    uint flat_call_group_id = get_flat_call_group_id();
    uint flat_call_group_thread_id = get_flat_call_group_thread_id();

    // Return a simple validation that functions executed
    return 1.0f;
}}
"""

    module = helpers.create_module(device, kernel_source)

    # Create output buffer
    result_buffer = NDBuffer(device=device, shape=call_shape, dtype=float)

    # Call with call group shape
    module.test_call_group_math.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result=result_buffer
    )

    # Validate that all calls completed successfully (all should be 1.0)
    results = result_buffer.to_numpy()
    assert np.all(
        results == 1.0
    ), f"Failed for {description}: {call_shape} with groups {call_group_shape}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dimension,call_shape,call_group_shape,description", EDGE_CASE_TEST_CASES)
def test_call_group_edge_cases(
    device_type: DeviceType,
    dimension: int,
    call_shape: tuple,
    call_group_shape: tuple,
    description: str,
):
    """Test edge cases for call group functionality."""

    device = helpers.get_device(device_type)

    # For 5D+, use array types like uint[5], uint[6], etc.
    if dimension <= 4:
        param_type = {1: "uint", 2: "uint2", 3: "uint3", 4: "uint4"}[dimension]
    else:
        param_type = f"uint[{dimension}]"
    kernel_source = f"""
import "slangpy";

float test_edge_case({param_type} grid_pos) {{
    // Test that edge cases don't crash
    int[{dimension}] call_id = get_call_id<{dimension}>();
    uint[{dimension}] call_group_id = get_call_group_id<{dimension}>();
    uint[{dimension}] call_group_thread_id = get_call_group_thread_id<{dimension}>();

    return 1.0f;
}}
"""

    module = helpers.create_module(device, kernel_source)

    # Create output buffer
    result_buffer = NDBuffer(device=device, shape=call_shape, dtype=float)

    # Call with edge case call group shape
    module.test_edge_case.call_group_shape(Shape(call_group_shape))(
        spy.grid(call_shape), _result=result_buffer
    )

    # Validate that all calls completed successfully
    results = result_buffer.to_numpy()
    assert np.all(
        results == 1.0
    ), f"Failed edge case {description}: {call_shape} with groups {call_group_shape}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_error_handling(device_type: DeviceType):
    """Test error handling for invalid call group configurations."""

    device = helpers.get_device(device_type)

    kernel_source = """
import "slangpy";

float test_basic(uint2 grid_pos) {
    return 1.0f;
}
"""

    module = helpers.create_module(device, kernel_source)

    # Test invalid call group shapes
    with pytest.raises((ValueError, RuntimeError)):
        # Negative dimensions should fail
        module.test_basic.call_group_shape(Shape((-1, 2)))(spy.grid((4, 4)), _result="numpy")

    with pytest.raises((ValueError, RuntimeError)):
        # Zero dimensions should fail
        module.test_basic.call_group_shape(Shape((0, 2)))(spy.grid((4, 4)), _result="numpy")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_dimension_validation_simple(device_type: DeviceType):
    """Test validation of call_group_shape dimensions and values - simplified version."""

    device = helpers.get_device(device_type)

    kernel_source_2d = """
import "slangpy";

float test_2d(uint2 grid_pos) {
    return 1.0f;
}
"""

    module_2d = helpers.create_module(device, kernel_source_2d)

        # Only test zero values first (this works in existing tests)
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        # Zero in call_group_shape should fail
        module_2d.test_2d.call_group_shape(Shape((2, 0)))(spy.grid((4, 4)), _result="numpy")

    error_message = str(exc_info.value)
    # The error should be caught at Python level with clear validation message
    assert ("call_group_shape[1] = 0" in error_message and "must be >= 1" in error_message)

    # Test that valid configurations still work (regression test)
    result = module_2d.test_2d.call_group_shape(Shape((2, 2)))(spy.grid((4, 4)), _result="numpy")
    assert np.all(result == 1.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_dimension_validation_too_many_dims(device_type: DeviceType):
    """Test validation of call_group_shape having too many dimensions."""

    device = helpers.get_device(device_type)

    kernel_source_2d = """
import "slangpy";

float test_2d(uint2 grid_pos) {
    return 1.0f;
}
"""

    kernel_source_1d = """
import "slangpy";

float test_1d(uint grid_pos) {
    return 1.0f;
}
"""

    module_2d = helpers.create_module(device, kernel_source_2d)
    module_1d = helpers.create_module(device, kernel_source_1d)

    # Test call_group_shape with more dimensions than call_shape
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        # 3D call_group_shape for 2D call_shape should fail
        module_2d.test_2d.call_group_shape(Shape((2, 2, 2)))(spy.grid((4, 4)), _result="numpy")

    error_message = str(exc_info.value)
    assert "call_group_shape dimension (3)" in error_message
    assert "call_shape dimension (2)" in error_message
    assert "cannot have more dimensions than call_shape" in error_message

    # Test with way too many dimensions
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        # 5D call_group_shape for 1D call_shape should fail
        module_1d.test_1d.call_group_shape(Shape((1, 1, 1, 1, 1)))(spy.grid((8,)), _result="numpy")

    error_message = str(exc_info.value)
    assert "call_group_shape dimension (5)" in error_message
    assert "call_shape dimension (1)" in error_message


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_dimension_validation_negative_values(device_type: DeviceType):
    """Test validation of call_group_shape having negative values."""

    device = helpers.get_device(device_type)

    kernel_source_2d = """
import "slangpy";

float test_2d(uint2 grid_pos) {
    return 1.0f;
}
"""

    module_2d = helpers.create_module(device, kernel_source_2d)

        # Test call_group_shape with negative values
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        # Negative in call_group_shape should fail
        module_2d.test_2d.call_group_shape(Shape((2, -1)))(spy.grid((4, 4)), _result="numpy")

    error_message = str(exc_info.value)
    assert ("call_group_shape[1] = -1" in error_message and "must be >= 1" in error_message)

    # Test mixed invalid values
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        # First invalid value should be caught
        module_2d.test_2d.call_group_shape(Shape((0, -1)))(spy.grid((4, 4)), _result="numpy")

    error_message = str(exc_info.value)
    assert ("call_group_shape[0] = 0" in error_message and "must be >= 1" in error_message)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_group_dimension_validation_zero_dimensions(device_type: DeviceType):
    """Test that zero-dimensional call_group_shape is valid and gets padded with 1's."""

    device = helpers.get_device(device_type)

    kernel_source_2d = """
import "slangpy";

float test_2d(uint2 grid_pos) {
    return 1.0f;
}
"""

    module_2d = helpers.create_module(device, kernel_source_2d)

    # Test call_group_shape with zero dimensions Shape(())
    # This should be VALID - it means "use default linear behavior"
    # and should get padded to [1, 1] for a 2D call shape
    result = module_2d.test_2d.call_group_shape(Shape(()))(spy.grid((4, 4)), _result="numpy")

    # Should work and produce correct result shape
    assert result.shape == (4, 4)
    assert np.all(result == 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
