import collections.abc
from typing import Iterable

import numpy as np
from taichi_forge.lang import ops
from taichi_forge.lang.exception import TaichiSyntaxError, TaichiTypeError
from taichi_forge.lang.expr import Expr
from taichi_forge.lang.matrix import Matrix
from taichi_forge.types.utils import is_integral


class _Ndrange:
    def __init__(self, *args):
        args = list(args)
        for i, arg in enumerate(args):
            if not isinstance(arg, collections.abc.Sequence):
                args[i] = (0, arg)
            if len(args[i]) != 2:
                raise TaichiSyntaxError(
                    "Every argument of ndrange should be a scalar or a tuple/list like (begin, end)"
                )
            args[i] = (args[i][0], ops.max(args[i][0], args[i][1]))
        for arg in args:
            for bound in arg:
                if not isinstance(bound, (int, np.integer)) and not (
                    isinstance(bound, Expr) and is_integral(bound.ptr.get_rvalue_type())
                ):
                    raise TaichiTypeError(
                        "Every argument of ndrange should be an integer scalar or a tuple/list of (int, int)"
                    )
        self.bounds = args

        self.dimensions = [None] * len(args)
        for i, bound in enumerate(self.bounds):
            self.dimensions[i] = bound[1] - bound[0]

        self.acc_dimensions = self.dimensions.copy()
        for i in reversed(range(len(self.bounds) - 1)):
            self.acc_dimensions[i] = self.acc_dimensions[i] * self.acc_dimensions[i + 1]
        if len(self.acc_dimensions) == 0:  # for the empty case, e.g. ti.ndrange()
            self.acc_dimensions = [1]

    def __iter__(self):
        def gen(d, prefix):
            if d == len(self.bounds):
                yield prefix
            else:
                for t in range(self.bounds[d][0], self.bounds[d][1]):
                    yield from gen(d + 1, prefix + (t,))

        yield from gen(0, ())

    def grouped(self):
        return GroupedNDRange(self)


def ndrange(*args) -> Iterable:
    """Return an immutable iterator object for looping over multi-dimensional indices.

    This returned set of multi-dimensional indices is the direct product (in the set-theory sense)
    of n groups of integers, where n equals the number of arguments in the input list, and looks like

    range(x1, y1) x range(x2, y2) x ... x range(xn, yn)

    The k-th argument corresponds to the k-th `range()` factor in the above product, and each
    argument must be an integer or a pair of two integers. An integer argument n will be interpreted
    as `range(0, n)`, and a pair of two integers (start, end) will be interpreted as `range(start, end)`.

    You can loop over these multi-dimensonal indices in different ways, see the examples below.

    Args:
        entries: (int, tuple): Must be either an integer, or a tuple/list of two integers.

    Returns:
        An immutable iterator object.

    Example::

        You can loop over 1-D integers in range [start, end), as in native Python

            >>> @ti.kernel
            >>> def loop_1d():
            >>>     start = 2
            >>>     end = 5
            >>>     for i in ti.ndrange((start, end)):
            >>>         print(i)  # will print 2 3 4

        Note the braces around `(start, end)` in the above code. If without them,
        the parameter `2` will be interpreted as `range(0, 2)`, `5` will be
        interpreted as `range(0, 5)`, and you will get a set of 2-D indices which
        contains 2x5=10 elements, and need two indices i, j to loop over them:

            >>> @ti.kernel
            >>> def loop_2d():
            >>>     for i, j in ti.ndrange(2, 5):
            >>>         print(i, j)
            0 0
            ...
            0 4
            ...
            1 4

        A kernel ndrange loop needs one loop variable per dimension. Use
        ti.grouped to collect all coordinates in one vector variable:

            >>> @ti.kernel
            >>> def loop_2d_grouped():
            >>>     for I in ti.grouped(ti.ndrange(2, 5)):
            >>>         print(I[0], I[1])

        For a flat index use range(2 * 5), then decode row and column with
        i // 5 and i % 5. Parallel execution does not guarantee print order.

        A typical usage of `ndrange` is when you want to loop over a tensor and process
        its entries in parallel. You should avoid writing nested `for` loops here since
        only top level `for` loops are paralleled in taichi, instead you can use `ndrange`
        to hold all entries in one top level loop:

            >>> @ti.kernel
            >>> def loop_tensor():
            >>>     for row, col, channel in ti.ndrange(image_height, image_width, channels):
            >>>         image[row, col, channel] = ...
    """
    return _Ndrange(*args)


class GroupedNDRange:
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        for ind in self.r:
            yield Matrix(list(ind))


__all__ = ["ndrange"]
