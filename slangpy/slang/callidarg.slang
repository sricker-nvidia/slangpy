// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct CallIdArg {

    // Array load
    public void load<let N : int, T: __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_id;
        for (int i = 0; i < N; i++) {
            value[i] = (T)t[i];
        }
    }

    // Vector load (transpose of array load)
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out vector<T,N> value) {
        var t = context.call_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end-i] = (T)t[i];
        }
    }

    // Scalar load (works for N = 1)
    public void load<T : __BuiltinIntegerType>(ContextND<1> context, out T value) {
        value = (T)context.call_id[0];
    }
}

// Rules for how to vectorize a Python call id argument to a given Slang type.
// Dim is user specified Python side, -1 means use the same dimensionality as the Slang type.
struct VectorizeCallidArgTo<SlangParameterType, let Dim: int> {
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<Array<T,N>, N> {
    typealias VectorType = Array<T,N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<Array<T, N>, -1>  {
    typealias VectorType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<vector<T,N>, N> {
    typealias VectorType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallidArgTo<vector<T, N>, -1> {
    typealias VectorType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> VectorizeCallidArgTo<T, -1> {
    typealias VectorType = T;
}
extension<T : __BuiltinIntegerType> VectorizeCallidArgTo<T, 1> {
    typealias VectorType = T;
}

public struct CallGroupIdArg {

    // Array load
    public void load<let N : int, T: __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_group_id;
        for (int i = 0; i < N; i++) {
            value[i] = (T)t[i];
        }
    }

    // Vector load (transpose of array load)
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out vector<T,N> value) {
        var t = context.call_group_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end-i] = (T)t[i];
        }
    }

    // Scalar load (works for N = 1)
    public void load<T : __BuiltinIntegerType>(ContextND<1> context, out T value) {
        value = (T)context.call_group_id[0];
    }
}

public struct CallGroupThreadIdArg {

    // Array load
    public void load<let N : int, T: __BuiltinIntegerType>(ContextND<N> context, out Array<T, N> value) {
        var t = context.call_group_thread_id;
        for (int i = 0; i < N; i++) {
            value[i] = (T)t[i];
        }
    }

    // Vector load (transpose of array load)
    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out vector<T,N> value) {
        var t = context.call_group_thread_id;
        int end = N - 1;
        for (int i = 0; i < N; i++) {
            value[end-i] = (T)t[i];
        }
    }

    // Scalar load (works for N = 1)
    public void load<T : __BuiltinIntegerType>(ContextND<1> context, out T value) {
        value = (T)context.call_group_thread_id[0];
    }
}

public struct FlatCallGroupIdArg {

    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out T value) {
        value = (T)context.flat_call_group_id;
    }
}

public struct FlatCallGroupThreadIdArg {

    public void load<let N : int, T : __BuiltinIntegerType>(ContextND<N> context, out T value) {
        value = (T)context.flat_call_group_thread_id;
    }
}

// Rules for how to vectorize a Python call group id argument to a given Slang type.
// Dim is user specified Python side, -1 means use the same dimensionality as the Slang type.
struct VectorizeCallGroupIdArgTo<SlangParameterType, let Dim: int> {
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupIdArgTo<Array<T,N>, N> {
    typealias VectorType = Array<T,N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupIdArgTo<Array<T, N>, -1>  {
    typealias VectorType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupIdArgTo<vector<T,N>, N> {
    typealias VectorType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupIdArgTo<vector<T, N>, -1> {
    typealias VectorType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> VectorizeCallGroupIdArgTo<T, -1> {
    typealias VectorType = T;
}
extension<T : __BuiltinIntegerType> VectorizeCallGroupIdArgTo<T, 1> {
    typealias VectorType = T;
}

// Rules for how to vectorize a Python call group thread id argument to a given Slang type.
struct VectorizeCallGroupThreadIdArgTo<SlangParameterType, let Dim: int> {
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupThreadIdArgTo<Array<T,N>, N> {
    typealias VectorType = Array<T,N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupThreadIdArgTo<Array<T, N>, -1>  {
    typealias VectorType = Array<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupThreadIdArgTo<vector<T,N>, N> {
    typealias VectorType = vector<T, N>;
}
extension<let N : int, T : __BuiltinIntegerType> VectorizeCallGroupThreadIdArgTo<vector<T, N>, -1> {
    typealias VectorType = vector<T, N>;
}
extension<T : __BuiltinIntegerType> VectorizeCallGroupThreadIdArgTo<T, -1> {
    typealias VectorType = T;
}
extension<T : __BuiltinIntegerType> VectorizeCallGroupThreadIdArgTo<T, 1> {
    typealias VectorType = T;
}

// Functions provided for use in user's shaders
export public int[N] get_call_id<let N: int>();
export public uint[N] get_call_group_id<let N: int>();
export public uint[N] get_call_group_thread_id<let N: int>();

export public int get_flat_call_id();
export public uint get_flat_call_group_id();
export public uint get_flat_call_group_thread_id();
