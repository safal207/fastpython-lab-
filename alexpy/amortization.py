"""Opt-in, exact-int list-to-buffer amortization experiment.

The public AlexPy dispatcher is never changed.  This module deliberately
rejects exotic objects rather than bypassing custom Python iteration hooks.
Measured workflows include allocation, packing and optional unpacking.
"""
from __future__ import annotations

from array import array

from . import sum_squares
from .memory_candidates import _buffer_native

_MIN_Q = -(1 << 63)
_MAX_Q = (1 << 63) - 1


def _arguments(repetitions: int, materialize: bool) -> None:
    if type(repetitions) is not int:
        raise TypeError('repetitions must be an exact int')
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    if type(materialize) is not bool:
        raise TypeError('materialize must be a bool')


def _pack(values):
    """Return packed q array or NotImplemented without consuming custom hooks.

    Values are not mutated during a measurement; only exact builtins are
    permitted.  The O(N) scan is part of the measured conversion cost.
    """
    if _buffer_native is None or type(values) not in (list, tuple):
        return NotImplemented
    if array('q').itemsize != 8:
        return NotImplemented
    for value in values:
        if type(value) is not int or not _MIN_Q <= value <= _MAX_Q:
            return NotImplemented
    return array('q', values)


def reference_batch(values, repetitions: int, *, materialize: bool = False):
    """K evaluations on the original list; optionally return a new list copy."""
    _arguments(repetitions, materialize)
    total = 0
    for _ in range(repetitions):
        total += sum_squares(values)
    return (total, list(values)) if materialize else total


def packed_batch(values, repetitions: int, *, materialize: bool = False):
    """Pack once, reuse K times, and optionally reconstruct a list.

    Returns NotImplemented for unsupported inputs. Not safe on concurrent
    source mutation; the source must remain unchanged during the batch.
    """
    _arguments(repetitions, materialize)
    packed = _pack(values)
    if packed is NotImplemented:
        return NotImplemented
    total = 0
    for _ in range(repetitions):
        total += _buffer_native.sum_squares_linear(packed)
    return (total, list(packed)) if materialize else total


def repacked_batch(values, repetitions: int, *, materialize: bool = False):
    """Negative control: conversion before EVERY evaluation, same output."""
    _arguments(repetitions, materialize)
    packed = _pack(values)
    if packed is NotImplemented:
        return NotImplemented
    total = 0
    for step in range(repetitions):
        if step:
            packed = _pack(values)
        total += _buffer_native.sum_squares_linear(packed)
    return (total, list(packed)) if materialize else total


__all__ = ['reference_batch', 'packed_batch', 'repacked_batch']
