# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from slangpy.bindings import (
    PYTHON_TYPES,
    AccessType,
    Marshall,
    BindContext,
    BoundVariable,
    CodeGenBlock,
)
from slangpy.experimental.gridarg import grid
from slangpy.reflection import SlangProgramLayout, SlangType
from slangpy.bindings import Shape


class CallIdArg:
    """
    Passes the thread id as an argument to a SlangPy function.
    """

    def __init__(self, dims: int = -1):
        super().__init__()
        if isinstance(dims, tuple):
            raise ValueError(
                "Using call id argument with a tuple is deprecated. To specify a shape, use the 'grid' argument type instead of 'call_id'"
            )
        self.dims = dims

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


def call_id(dims: int = -1):
    """
    Create a ThreadIdArg to pass to a SlangPy function, which passes the thread id.
    """
    if isinstance(dims, tuple):
        return grid(shape=dims)
    return CallIdArg(dims)


class CallIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"CallIdArg")
        if st is None:
            raise ValueError(
                f"Could not find CallIdArg slang type. This usually indicates the threadidarg module has not been imported."
            )
        self.slang_type = st
        self.match_call_shape = True

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        # Resolve type using reflection.
        conv_type = bound_type.program.find_type_by_name(
            f"VectorizeCallidArgTo<{bound_type.full_name}, {self.dims}>.VectorType"
        )
        if conv_type is None:
            raise ValueError(
                f"Could not find suitable conversion from CallIdArg<{self.dims}> to {bound_type.full_name}"
            )
        return conv_type

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        # Call id arg is generated for every thread and has no effect on call shape,
        # so it returns -1 to indicate it doesn't contribute to dimensionality.
        return -1


PYTHON_TYPES[CallIdArg] = lambda l, x: CallIdArgMarshall(l, x.dims)


class CallGroupIdArg:
    """
    Passes the call group id as an argument to a SlangPy function.
    """

    def __init__(self, dims: int = -1):
        super().__init__()
        self.dims = dims

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


class CallGroupThreadIdArg:
    """
    Passes the call group thread id as an argument to a SlangPy function.
    """

    def __init__(self, dims: int = -1):
        super().__init__()
        self.dims = dims

    @property
    def slangpy_signature(self) -> str:
        return f"[{self.dims}]"


class FlatCallGroupIdArg:
    """
    Passes the flat call group id as an argument to a SlangPy function.
    """

    def __init__(self):
        super().__init__()

    @property
    def slangpy_signature(self) -> str:
        return "[]"


class FlatCallGroupThreadIdArg:
    """
    Passes the flat call group thread id as an argument to a SlangPy function.
    """

    def __init__(self):
        super().__init__()

    @property
    def slangpy_signature(self) -> str:
        return "[]"


def call_group_id(dims: int = -1):
    """
    Create a CallGroupIdArg to pass to a SlangPy function, which passes the call group id.
    """
    return CallGroupIdArg(dims)


def call_group_thread_id(dims: int = -1):
    """
    Create a CallGroupThreadIdArg to pass to a SlangPy function, which passes the call group thread id.
    """
    return CallGroupThreadIdArg(dims)


def flat_call_group_id():
    """
    Create a FlatCallGroupIdArg to pass to a SlangPy function, which passes the flat call group id.
    """
    return FlatCallGroupIdArg()


def flat_call_group_thread_id():
    """
    Create a FlatCallGroupThreadIdArg to pass to a SlangPy function, which passes the flat call group thread id.
    """
    return FlatCallGroupThreadIdArg()


class CallGroupIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"CallGroupIdArg")
        if st is None:
            raise ValueError(
                f"Could not find CallGroupIdArg slang type. This usually indicates the callidarg module has not been imported."
            )
        self.slang_type = st
        self.match_call_shape = True

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        # Resolve type using reflection.
        conv_type = bound_type.program.find_type_by_name(
            f"VectorizeCallGroupIdArgTo<{bound_type.full_name}, {self.dims}>.VectorType"
        )
        if conv_type is None:
            raise ValueError(
                f"Could not find suitable conversion from CallGroupIdArg<{self.dims}> to {bound_type.full_name}"
            )
        return conv_type

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        # Call group id arg is generated for every thread and has no effect on call shape,
        # so it returns -1 to indicate it doesn't contribute to dimensionality.
        return -1


class CallGroupThreadIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout, dims: int):
        super().__init__(layout)
        self.dims = dims
        st = layout.find_type_by_name(f"CallGroupThreadIdArg")
        if st is None:
            raise ValueError(
                f"Could not find CallGroupThreadIdArg slang type. This usually indicates the callidarg module has not been imported."
            )
        self.slang_type = st
        self.match_call_shape = True

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        # Resolve type using reflection.
        conv_type = bound_type.program.find_type_by_name(
            f"VectorizeCallGroupThreadIdArgTo<{bound_type.full_name}, {self.dims}>.VectorType"
        )
        if conv_type is None:
            raise ValueError(
                f"Could not find suitable conversion from CallGroupThreadIdArg<{self.dims}> to {bound_type.full_name}"
            )
        return conv_type

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        # Call group thread id arg is generated for every thread and has no effect on call shape,
        # so it returns -1 to indicate it doesn't contribute to dimensionality.
        return -1


class FlatCallGroupIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout):
        super().__init__(layout)
        st = layout.find_type_by_name(f"FlatCallGroupIdArg")
        if st is None:
            raise ValueError(
                f"Could not find FlatCallGroupIdArg slang type. This usually indicates the callidarg module has not been imported."
            )
        self.slang_type = st
        self.concrete_shape = Shape(0)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        # Flat call group id arg should map to uint type
        return bound_type.program.find_type_by_name("uint")

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        return 0


class FlatCallGroupThreadIdArgMarshall(Marshall):
    def __init__(self, layout: SlangProgramLayout):
        super().__init__(layout)
        st = layout.find_type_by_name(f"FlatCallGroupThreadIdArg")
        if st is None:
            raise ValueError(
                f"Could not find FlatCallGroupThreadIdArg slang type. This usually indicates the callidarg module has not been imported."
            )
        self.slang_type = st
        self.concrete_shape = Shape(0)

    def gen_calldata(self, cgb: CodeGenBlock, context: BindContext, binding: BoundVariable):
        access = binding.access
        name = binding.variable_name
        if access[0] == AccessType.read:
            cgb.type_alias(f"_t_{name}", self.slang_type.full_name)

    def resolve_type(self, context: BindContext, bound_type: "SlangType"):
        # Flat call group thread id arg should map to uint type
        return bound_type.program.find_type_by_name("uint")

    def resolve_dimensionality(
        self,
        context: BindContext,
        binding: BoundVariable,
        vector_target_type: "SlangType",
    ):
        return 0


# Register the call group types in the PYTHON_TYPES registry
PYTHON_TYPES[CallGroupIdArg] = lambda l, x: CallGroupIdArgMarshall(l, x.dims)
PYTHON_TYPES[CallGroupThreadIdArg] = lambda l, x: CallGroupThreadIdArgMarshall(l, x.dims)
PYTHON_TYPES[FlatCallGroupIdArg] = lambda l, x: FlatCallGroupIdArgMarshall(l)
PYTHON_TYPES[FlatCallGroupThreadIdArg] = lambda l, x: FlatCallGroupThreadIdArgMarshall(l)
