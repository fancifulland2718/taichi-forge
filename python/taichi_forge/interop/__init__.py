"""Zero-copy storage interoperability.

The public adapters in this module are strict: unsupported storage raises
instead of silently materializing a copy. Existing NumPy, PyTorch, and Paddle
kernel-argument APIs retain their historical copy fallbacks.
"""

from ._dlpack import ExternalDenseView, capabilities, from_dlpack

__all__ = ["ExternalDenseView", "capabilities", "from_dlpack"]
