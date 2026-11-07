"""Synchronous Host buffer calls through the existing native array transfers."""

import ctypes
import sys

from ._runtime import NdArray


class HostBuffers:
    """Stage disjoint explicit Host arrays, without changing device calls."""

    def __init__(self, runtime, schemas, arguments):
        self.runtime, self.arguments = runtime, list(arguments)
        self.pending, self.allocated = [], []
        for index, (schema, value) in enumerate(zip(schemas, arguments)):
            if schema["kind"] != "ndarray" or isinstance(value, NdArray):
                continue
            try:
                view = memoryview(value)
            except TypeError:
                raise TypeError("Package array arguments require a native array or typed Host buffer") from None
            element = schema["element"]
            if element["kind"] == "struct":
                raise TypeError("Structured Host buffers require an explicit native StructLayout array")
            if not view.c_contiguous or view.readonly:
                raise ValueError("Package Host arrays require writable C-contiguous buffers")
            format = view.format
            if format[:1] in ("<", ">", "!"):
                if (format[0] == "<") != (sys.byteorder == "little"):
                    raise ValueError("Package Host array byte order differs from the Runtime")
                format = format[1:]
            format = format.lstrip("@=")
            category = "f" if format in ("f", "d") else "i" if format in ("b", "h", "i", "l", "q") else "u"
            if format not in ("f", "d", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"):
                raise TypeError("Unsupported Host array buffer format: " + view.format)
            if category + str(view.itemsize * 8) != element["dtype"]:
                raise ValueError("Host array dtype differs from the compiled entry")
            element_shape = tuple(element["element_shape"])
            if view.ndim != schema["ndim"] + len(element_shape):
                raise ValueError("Host array rank differs from the compiled entry")
            if element_shape and view.shape[-len(element_shape) :] != element_shape:
                raise ValueError("Host array element shape differs from the compiled entry")
            shape = view.shape[: schema["ndim"]]
            start = ctypes.addressof(ctypes.c_ubyte.from_buffer(view)) if view.nbytes else 0
            end = start + view.nbytes
            signature = (element["dtype"], shape, element_shape)
            for item in self.pending:
                if start < item["end"] and item["start"] < end:
                    raise ValueError(
                        "Aliased or overlapping Host arrays require an explicit compatible resource contract"
                    )
            self.pending.append(dict(start=start, end=end, signature=signature, view=view, index=index))

    def __enter__(self):
        try:
            for item in self.pending:
                dtype, shape, element_shape = item["signature"]
                array = self.runtime.ndarray(dtype, shape, element_shape=element_shape)
                self.allocated.append((array, item["view"]))
                array.write(item["view"])
                self.arguments[item["index"]] = array
            return self.arguments
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None:
                for array, view in self.allocated:
                    array.read_into(view)
        finally:
            for array, _ in reversed(self.allocated):
                array.close()
