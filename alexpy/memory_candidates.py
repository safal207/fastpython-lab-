"""Opt-in data-layout interventions, never installed as public defaults.

Only an exact array.array('q') or exact one-dimensional native-q memoryview
is passed to the zero-copy C kernels. Every unsupported object follows the
original Python semantics through ``alexpy.sum_squares``.
"""
from __future__ import annotations

from array import array
from . import sum_squares

try:
    from . import _buffer_native
except ImportError:
    _buffer_native = None


def eligible(values) -> bool:
    """Eligibility is a *representation* check, not an optimization claim."""
    if type(values) is array:
        return values.typecode == "q"
    if type(values) is memoryview:
        return (values.ndim == 1 and values.format == "q" and
                values.itemsize == 8 and values.c_contiguous)
    return False


def linear_buffer(values):
    if _buffer_native is not None and eligible(values):
        value = _buffer_native.sum_squares_linear(values)
        if value is not NotImplemented:
            return value
    return sum_squares(values)


def interleaved_buffer(values):
    if _buffer_native is not None and eligible(values):
        value = _buffer_native.sum_squares_interleaved(values)
        if value is not NotImplemented:
            return value
    return sum_squares(values)


def list_conversion(values):
    """Honest layout-conversion control: copy is inside the timed callable.

    Exact q arrays only; never silently alter iteration of subclasses.
    """
    if eligible(values):
        return sum_squares(list(values))
    return sum_squares(values)


CANDIDATES = {
    "current": sum_squares,
    "linear": linear_buffer,
    "interleaved64": interleaved_buffer,
    "convert_list": list_conversion,
}

__all__ = ["eligible", "linear_buffer", "interleaved_buffer",
           "list_conversion", "CANDIDATES"]
