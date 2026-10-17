"""Immutable AOT image tables use the shared resource registry and owners."""

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_texture_collection_resource_capability_and_ownership():
    with ti.aot.borrow_runtime() as runtime:
        available = runtime.native_resource_capabilities("sampled_image_collection")
        if ti.cfg.arch != ti.vulkan:
            assert not available
            return
        assert available

        def image(shape):
            return runtime.native_resource("image", "sampled", parameters=dict(shape=shape, format="r32f"))

        def table(members, dimensions=2):
            return runtime.native_resource(
                "image",
                "collection",
                parameters=dict(dimensions=dimensions, image_count=len(members)),
                bindings={f"image_{i}": member for i, member in enumerate(members)},
            )

        first, second, wrong = image([2, 2]), image([3, 2]), image([2])
        collection = table([second, first, second])
        with pytest.raises(RuntimeError, match="dimensionality"):
            table([first, wrong])
        with pytest.raises(RuntimeError, match="capacity"):
            table([])
        with ti.aot.borrow_runtime() as other:
            foreign = other.native_resource("image", "sampled", parameters=dict(shape=[2, 2], format="r32f"))
            with pytest.raises(ValueError, match="another runtime"):
                table([first, foreign])
        # Closing a member retires its dependent table. Explicit child close
        # remains safe even while its Python wrapper outlives native retirement.
        first.close()
        collection.close()
        collection.close()
        second.close()
        wrong.close()
