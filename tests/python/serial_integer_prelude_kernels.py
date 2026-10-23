"""Independent integer-interval and snapshot fixtures for compiler regression tests."""

import taichi_forge as ti


_U64_SENTINEL = 0xFFFFFFFFFFFFFFFF


@ti.data_oriented
class IntervalFixture:
    def __init__(self, capacity):
        self.capacity = capacity
        self.active_count = ti.field(ti.i32, shape=())
        self.lower_bound = ti.field(ti.u64, shape=capacity)
        self.upper_bound = ti.field(ti.u64, shape=capacity)
        self.status = ti.field(ti.i32, shape=capacity)

    @ti.kernel
    def classify(self):
        count = self.active_count[None]
        for index in range(self.capacity):
            if index < count:
                lower = self.lower_bound[index]
                upper = self.upper_bound[index]
                status = 0
                if lower != ti.u64(_U64_SENTINEL):
                    status = 1
                    if lower < upper:
                        status = 2
                self.status[index] = status

    @ti.kernel
    def seed(self, count: ti.i32,
             lower: ti.types.ndarray(ti.u64, 1),
             upper: ti.types.ndarray(ti.u64, 1)):
        self.active_count[None] = count
        for i in range(self.capacity):
            self.lower_bound[i] = lower[i]
            self.upper_bound[i] = upper[i]
            self.status[i] = -91

    @ti.kernel
    def report(self, out: ti.types.ndarray(ti.i32, 1)):
        for i in range(self.capacity):
            out[i] = self.status[i]


def make_guards(capacity):
    source = ti.field(ti.i32, shape=())
    out = ti.field(ti.i32, shape=capacity)
    x = ti.field(ti.f32, shape=capacity, needs_grad=True, needs_dual=True)
    y = ti.field(ti.f32, shape=capacity, needs_grad=True, needs_dual=True)

    @ti.kernel
    def dynamic_range():
        end = source[None]
        for i in range(end):
            out[i] = i

    @ti.kernel
    def source_write():
        snapshot = source[None]
        for i in range(capacity):
            out[i] = snapshot
            if i == 0:
                source[None] = 3

    @ti.kernel
    def external_alias(alias: ti.types.ndarray(ti.i32, 0)):
        snapshot = source[None]
        for i in range(capacity):
            out[i] = snapshot
            if i == 0:
                alias[None] = 3

    @ti.kernel
    def forward_prefix():
        count = source[None]
        for i in range(capacity):
            if i < count:
                y[i] = x[i] * x[i]

    @ti.kernel
    def reverse_dynamic():
        # Reverse AD rejects a user assignment before a top-level loop. This
        # legal frontend expression still requires a device-snapshot bound.
        for i in range(source[None]):
            y[i] = x[i] * x[i]

    return dict(source=source, out=out, x=x, y=y,
                dynamic_range=dynamic_range, source_write=source_write,
                external_alias=external_alias, forward_prefix=forward_prefix,
                reverse_dynamic=reverse_dynamic)
