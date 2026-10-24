"""Compatibility import for the common retained-helper lowering."""

from taichi_forge.graph._lowering_calls import lower_array_template_call as export_array_template_call

__all__ = ["export_array_template_call"]
