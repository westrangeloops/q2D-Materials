"""
Templates module for perovskite geometry definitions.

This module provides pure geometry templates that define fractional positions
for A, B, and X sites before any chemical species are assigned.
"""

from .templates import Template, CubicTemplate, ReducedTemplate

__all__ = ['Template', 'CubicTemplate', 'ReducedTemplate']

