"""
This module defines data types in Taichi:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.taichi_forge.graphics/publication/2021-quantaichi/quantaichi.pdf"
"""

from taichi_forge.types import quant
from taichi_forge.types.annotations import *
from taichi_forge.types.compound_types import *
from taichi_forge.types.ndarray_type import *
from taichi_forge.types.primitive_types import *
from taichi_forge.types.texture_type import *
from taichi_forge.types.utils import *
