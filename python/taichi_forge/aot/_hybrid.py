"""Cold, cached JIT storage imports for definition-package calls.

Type knowledge stays in the frontend. Native borrowed views retain the existing
Program, layout, alias and retirement checks; no data is copied or reallocated.
"""

import threading
import weakref

from taichi_forge.lang._ndarray import Ndarray, StructNdarray
from taichi_forge.lang._storage_view import DenseNdarrayView, ndarray_view
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField


class PackageArrays:
    def __init__(self, runtime):
        self.runtime = runtime
        self._cache = {}
        self._lock = threading.RLock()

    def __call__(self, value):
        if not isinstance(value, (Ndarray, DenseNdarrayView, ScalarField, MatrixField)):
            return value
        if isinstance(value, StructNdarray):
            raise TypeError("Hybrid package inputs require scalar or tensor storage")
        with self._lock:
            key = id(value)
            previous = self._cache.get(key)
            if previous is not None and previous[0]() is value:
                previous[1]._descriptor(self.runtime)
                return previous[1]
            if isinstance(value, Ndarray):
                imported = self.runtime.import_ndarray(value)
            else:
                view = value if isinstance(value, DenseNdarrayView) else ndarray_view(value)
                imported = self.runtime.import_dense_view(view)
            owner = weakref.ref(self)

            def release(reference):
                cache = owner()
                if cache is not None:
                    with cache._lock:
                        current = cache._cache.get(key)
                        if current is not None and current[0] is reference:
                            del cache._cache[key]

            self._cache[key] = (weakref.ref(value, release), imported)
            return imported

    def close(self):
        with self._lock:
            if getattr(self.runtime, "_package_array_adapter", None) is self:
                self.runtime._package_array_adapter = None
            # Prepared bindings hold their own view/lease references. Clearing
            # discovery does not revoke an independently retained invocation.
            self._cache.clear()
