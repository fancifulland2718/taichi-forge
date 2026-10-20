# Local Vulkan owner repair for CiG

These are optional, independently checked integration patches for the Vulkan
owner used in the [local validation report](../../../docs/forge/cuda_vulkan_cig_queue_validation.en.md).
They are not linked into Forge and introduce no Forge dependency on Flora,
Donut or NVRHI.

| Patch | Repository root | Purpose |
| --- | --- | --- |
| `flora-queue-routing.patch` | Matching Flora source | Explicit graphics/compute phases, actual publication queue, ordinary fallback |
| `nvrhi-queue-sharing.patch` | Its `external/donut/nvrhi` | Internal concurrent resources, queue-supported scopes, submitted query lifetime |
| `donut-queue-initialization.patch` | Its `external/donut` | Zero-initialize optional native queue handles |

The Flora preimage is `9251c23f` plus the existing GeoPhys lifetime/shader
candidate and prior CiG/ready integration. It is not an unmodified upstream
checkout. `SOURCE-MANIFEST.json` records SHA256 of normalized UTF-8 text for every
preimage and postimage, and the exact tested native. The local delivery also
contains these source images, the earlier ready integration patch and binaries.
Check these preconditions before applying to a different worktree.

From each matching repository, run `git apply --check <patch>` and then
`git apply <patch>`. All three patches are required. The local validation build
compiled isolated dependency copies through a temporary CMake source override;
that override is unnecessary when the patches are applied to the actual source
repositories. No shared dependency checkout was modified during the investigation.

Rebuild the renderer and run both ordinary CUDA and actual `vulkan_cig` controls.
The latter must reserve the external queue, retain the graphics preferred queue,
and enable a dedicated Vulkan compute family. Do not replace readiness admission
or the binary ready/consumed protocol with host timing assumptions.

The queue-initialization patch matters independently of CiG: the dependency's
descriptor has optional raw `VkQueue` members. Default initialization left them
indeterminate while their family indices were `-1`. The resulting phantom queue
caused invalid concurrent-sharing indices and command-pool creation. Value
initialization fixes the source of that fault; a capability query alone does not.

License notices for each underlying project are included alongside the patches.
