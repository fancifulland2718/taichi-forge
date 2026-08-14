import inspect

from scripts.validate_runtime_load_order import _make_fill_kernel


def test_fill_kernel_uses_materialized_ndarray_annotation():
    annotation = object()

    class FakeTypes:
        @staticmethod
        def ndarray(*, dtype, ndim):
            assert dtype is FakeTi.i32
            assert ndim == 1
            return annotation

    class FakeTi:
        i32 = object()
        types = FakeTypes()

        @staticmethod
        def kernel(function):
            parameter = inspect.signature(function).parameters["output"]
            assert parameter.annotation is annotation
            return function

    fill = _make_fill_kernel(FakeTi)

    assert fill.__name__ == "fill"
