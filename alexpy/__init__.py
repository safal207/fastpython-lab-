"""AlexPy: experimental native accelerators for specific Python operations.

Not a replacement for the CPython interpreter.
"""

from __future__ import annotations

try:
    from . import _native
except ImportError:
    _native = None

__version__ = "0.1.0"


def backend_name() -> str:
    """Name of the available execution backend."""
    return "native-c" if _native is not None else "python-fallback"


def sum_squares(values):
    """Return sum(x * x for x in values) with a safe native fast path.

    Exact list/tuple[int] inputs with a signed 64-bit result may execute in C.
    Other inputs fall back to Python, preserving arbitrary-precision integers,
    subclass behavior, Python's numerical protocols, and iterator semantics.
    """
    if _native is not None and type(values) in (list, tuple):
        native_result = _native.sum_squares_ints(values)
        if native_result is not NotImplemented:
            return native_result
    return sum(x * x for x in values)


__all__ = ["backend_name", "sum_squares", "__version__"]
