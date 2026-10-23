"""Same-Program borrowed sampled images keep one immutable read boundary.

Run on the paired candidate with Vulkan synchronization validation enabled.
These are integration tests; the admission-only CPU tests are not a substitute.
"""

from contextlib import ExitStack

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_graph_binding_frame_recipe import _catalog
from tests.python.test_vulkan_graph_execution_domain import domains


def fixture_graphs():
    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 2)):
        for x, y in output:
            output[x, y] = image.fetch(ti.Vector([x, y]), 0).x

    @ti.kernel
    def write(image: ti.types.rw_texture(2, fmt=ti.Format.r32f),
              values: ti.types.ndarray(ti.f32, 2)):
        for x, y in values:
            image.store(ti.Vector([x, y]), ti.Vector([values[x, y], 0, 0, 0]))

    sampled = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
    storage = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "writable", ndim=2, fmt=ti.Format.r32f)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=2)
    reader = ti.graph.GraphBuilder()
    reader.dispatch(sample, sampled, output)
    cycle = ti.graph.GraphBuilder()
    # Both entry and final layout are shader_read. The complete recording still
    # contains a storage write and must be blocked even without a layout repair.
    cycle.dispatch(sample, sampled, output)
    cycle.dispatch(write, storage, values)
    cycle.dispatch(sample, sampled, output)
    return sample, write, reader.compile(), cycle.freeze()


def prepared_writer(stack, definition, texture, values, output):
    catalog = _catalog(definition)
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    context = stack.enter_context(definition.materialization_context(provider_set=catalog.provider_set))
    materialized = stack.enter_context(context.materialize(recipe))
    graph = materialized.executor
    binding = graph.bind(dict(image=texture, writable=texture, values=values, output=output))
    assert binding._version.execution_frame.uses_secondary_commands()
    return graph, binding


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_two_shared_readers_block_prepared_writer_and_restore_same_recording():
    sample, write, reader, cycle = fixture_graphs()
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    values = ti.ndarray(ti.f32, (2, 2))
    expected = np.asarray([[2, 3], [5, 7]], np.float32)
    values.from_numpy(expected)
    texture.from_ndarray(values)
    outputs = [ti.ndarray(ti.f32, (2, 2)) for _ in range(3)]
    queues = domains(capacity=2, instance_capacity=2)
    calls = []
    try:
        with ExitStack() as stack:
            writer, binding = prepared_writer(stack, cycle, texture, values, outputs[2])
            writer.run(binding)
            ti.sync()
            frame = binding._version.execution_frame
            fixed_bytes = frame.argument_bytes()
            calls = [reader.prepare(dict(image=texture, output=output), execution_domain=queue)
                     for output, queue in zip(outputs, queues)]
            for call in calls:
                call.initialize()
            # Both domains borrow the same existing Texture lease generation;
            # observing the first completion does not consume the prefix.
            for call in calls:
                ticket = call.submit()
                ticket.wait()
                ticket.close()
            for output in outputs[:2]:
                np.testing.assert_array_equal(output.to_numpy(), expected)
            reader.run(dict(image=texture, output=outputs[2]))
            np.testing.assert_array_equal(outputs[2].to_numpy(), expected)

            values.fill(11)
            for operation in (lambda: texture.from_ndarray(values),
                              lambda: write(texture, values),
                              lambda: writer.run(binding)):
                with pytest.raises((ValueError, RuntimeError), match="read loan"):
                    operation()
            assert frame.argument_bytes() == fixed_bytes
            # Ordinary sampled dispatch remains legal after all rejections.
            sample(texture, outputs[2])
            np.testing.assert_array_equal(outputs[2].to_numpy(), expected)
            calls[0].close()
            with pytest.raises((ValueError, RuntimeError), match="read loan"):
                writer.run(binding)
            # A completed submit does not release the surviving instance loan.
            remaining = calls[1].submit()
            remaining.wait()
            remaining.close()
            with pytest.raises((ValueError, RuntimeError), match="read loan"):
                texture.from_ndarray(values)
            calls[1].close()
            # Reuse the exact previously blocked recording: no re-prepare.
            writer.run(binding)
            np.testing.assert_array_equal(outputs[2].to_numpy(), np.full((2, 2), 11, np.float32))
            assert frame.argument_bytes() == fixed_bytes
            texture.from_ndarray(values)
            replacement = reader.prepare(dict(image=texture, output=outputs[0]), execution_domain=queues[0])
            calls.append(replacement)
            replacement.initialize()
            reader.close()
            ticket = replacement.submit()
            ticket.wait()
            ticket.close()
            np.testing.assert_array_equal(outputs[0].to_numpy(), np.full((2, 2), 11, np.float32))
    finally:
        for call in calls:
            call.close()
        for queue in queues:
            queue.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_read_loan_survives_handle_release_until_domain_retirement():
    _, _, reader, _ = fixture_graphs()
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    source = ti.ndarray(ti.f32, (2, 2))
    output = ti.ndarray(ti.f32, (2, 2))
    source.fill(13)
    texture.from_ndarray(source)
    queues = domains(capacity=2, instance_capacity=1)
    call = reader.prepare(dict(image=texture, output=output), execution_domain=queues[0])
    call.initialize()
    reader.close()
    pending = call.submit()
    # release does not wait or retire the logical instance, even if the GPU
    # happens to complete this small dispatch before the following assertion.
    call.release()
    with pytest.raises((ValueError, RuntimeError), match="read loan"):
        texture.from_ndarray(source)
    queues[0].close()
    assert pending.done()
    np.testing.assert_array_equal(output.to_numpy(), np.full((2, 2), 13, np.float32))
    source.fill(17)
    texture.from_ndarray(source)
    queues[1].close()
    pending.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_failed_prepare_after_read_boundary_does_not_poison_image_or_domain():
    _, write, reader, _ = fixture_graphs()
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    source = ti.ndarray(ti.f32, (2, 2))
    output = ti.ndarray(ti.f32, (2, 2))
    source.fill(19)
    texture.from_ndarray(source)
    queues = domains(capacity=2, instance_capacity=1)
    # Reserving the image and publishing the read boundary precedes logical
    # instance admission. Fill the queue's one instance slot to fail afterward.
    @ti.kernel
    def ordinary(value: ti.types.ndarray(ti.f32, 2)):
        for x, y in value:
            value[x, y] += 0

    builder = ti.graph.GraphBuilder()
    builder.dispatch(ordinary, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "value", ti.f32, ndim=2))
    occupied_graph = builder.compile()
    occupied = occupied_graph.prepare(dict(value=output), execution_domain=queues[0])
    occupied.initialize()
    try:
        with pytest.raises(RuntimeError):
            reader.prepare(dict(image=texture, output=output), execution_domain=queues[0])
        # The prefix transition is real and must not be rolled back in the
        # tracker; no retained reservation may remain after failed preparation.
        source.fill(23)
        texture.from_ndarray(source)
        write(texture, source)
        healthy = reader.prepare(dict(image=texture, output=output), execution_domain=queues[1])
        healthy.initialize()
        ticket = healthy.submit()
        ticket.wait()
        ticket.close()
        np.testing.assert_array_equal(output.to_numpy(), np.full((2, 2), 23, np.float32))
        healthy.close()
    finally:
        occupied.close()
        for queue in queues:
            queue.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_initialized_reader_without_replay_releases_loan():
    _, _, reader, _ = fixture_graphs()
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    source = ti.ndarray(ti.f32, (2, 2))
    output = ti.ndarray(ti.f32, (2, 2))
    source.fill(37)
    texture.from_ndarray(source)
    queues = domains(capacity=2, instance_capacity=1)
    call = reader.prepare(dict(image=texture, output=output), execution_domain=queues[0])
    try:
        call.initialize()
        with pytest.raises((ValueError, RuntimeError), match="read loan"):
            texture.from_ndarray(source)
        call.close()
        # Initialization itself submitted Commands that retain the workspace.
        # Closing without any replay must collect that completed queue slot.
        texture.from_ndarray(source)
        reader.run(dict(image=texture, output=output))
        np.testing.assert_array_equal(output.to_numpy(), np.full((2, 2), 37, np.float32))
    finally:
        call.close()
        for queue in queues:
            queue.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_shared_image_rejects_multiple_mips_then_reuses_domain():
    _, _, reader, _ = fixture_graphs()
    source = ti.ndarray(ti.f32, (2, 2))
    output = ti.ndarray(ti.f32, (2, 2))
    source.fill(29)
    rejected = ti.Texture(ti.Format.r32f, (2, 2), mip_levels=2)
    accepted = ti.Texture(ti.Format.r32f, (2, 2))
    rejected.from_ndarray(source)
    accepted.from_ndarray(source)
    queues = domains(capacity=2, instance_capacity=1)
    try:
        with pytest.raises(RuntimeError, match="one mip and layer"):
            reader.prepare(dict(image=rejected, output=output), execution_domain=queues[0])
        call = reader.prepare(dict(image=accepted, output=output), execution_domain=queues[0])
        call.initialize()
        ticket = call.submit()
        ticket.wait()
        ticket.close()
        np.testing.assert_array_equal(output.to_numpy(), np.full((2, 2), 29, np.float32))
        call.close()
    finally:
        for queue in queues:
            queue.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_canvas_transfer_source_conflicts_with_shared_sampled_boundary():
    if not ti._lib.core.GGUI_AVAILABLE:
        pytest.skip("GGUI is not built")
    _, _, reader, _ = fixture_graphs()
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    source = ti.ndarray(ti.f32, (2, 2))
    output = ti.ndarray(ti.f32, (2, 2))
    source.fill(31)
    texture.from_ndarray(source)
    window = ti.ui.Window("shared image layout admission", (64, 64), show_window=False)
    queues = domains(capacity=2, instance_capacity=1)
    call = reader.prepare(dict(image=texture, output=output), execution_domain=queues[0])
    call.initialize()
    try:
        with pytest.raises((ValueError, RuntimeError), match="read loan"):
            window.get_canvas().set_image(texture)
        ticket = call.submit()
        ticket.wait()
        ticket.close()
        np.testing.assert_array_equal(output.to_numpy(), np.full((2, 2), 31, np.float32))
        call.close()
        window.get_canvas().set_image(texture)
        window.get_image_buffer_as_numpy()
    finally:
        call.close()
        for queue in queues:
            queue.close()
        window.destroy()
