#include "gtest/gtest.h"
#include "external/PicoSHA2/picosha2.h"

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/taichi_vulkan_impl.h"
#include "c_api/src/native_resource_registry.h"
#include "taichi/program/program.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/native_vulkan_ray_access.h"
#include "taichi/runtime/native_ray_admission.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/prepared_vulkan_ray_access.h"
#include "taichi/program/runtime_fault.h"
#include "tests/cpp/aot/vulkan/pending_domain_action.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

namespace taichi::lang {

namespace {
using namespace std::chrono_literals;

void require_success() {
  char message[1024]{};
  uint64_t size = sizeof(message);
  if (ti_get_last_error(&size, message) != TI_ERROR_SUCCESS)
    throw std::runtime_error(message);
}

struct BorrowScope {
  explicit BorrowScope(AotRuntimeClient &client) : client(client) {
    client.enter();
    ti_set_last_error(TI_ERROR_SUCCESS, "");
  }
  ~BorrowScope() {
    client.leave();
  }
  AotRuntimeClient &client;
};

struct PendingProgramAs {
  Program program{Arch::vulkan};
  std::array<std::shared_ptr<AotRuntimeClient>, 2> borrowers;
  std::shared_ptr<test_detail::GateOwner> gates =
      std::make_shared<test_detail::GateOwner>(3);
  std::array<TiRuntime, 2> runtimes{};
  std::array<TiAotModule, 2> modules{};
  std::array<TiVulkanExecutionDomain, 2> domains{};
  std::array<TiPreparedExecution, 2> executions{};
  std::array<TiCompletion, 2> completions{};
  std::array<TiMemory, 2> outputs{}, ids{};
  std::array<uint64_t, 2> scenes{};
  std::array<std::weak_ptr<NativeResource>, 2> observed;
  uint64_t blas{0};
  VulkanRayGeometryCommand blas_refit;
  std::array<VulkanTLASTransformCommand, 2> ordinary_refits;
  std::array<VulkanRayQueryCommand, 2> ordinary_queries;
  bool writes{false};

  PendingProgramAs() {
    program.materialize_runtime();
  }
  ~PendingProgramAs() {
    gates->release_all();
    program.finalize();
    for (auto ticket : completions)
      if (ticket)
        ti_release_completion_ext(ticket);
    for (auto domain : domains)
      if (domain)
        ti_release_vulkan_execution_domain_ext(domain);
  }
  vulkan::VulkanDevice &device() {
    return static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  }
  bool available() {
    const auto &caps = device().vk_caps();
    return caps.acceleration_structure && caps.ray_query &&
           caps.timeline_semaphore &&
           device().created_compute_queue_count() >= 3;
  }
  template <typename T, std::size_t N>
  void upload(Ndarray *array, const std::array<T, N> &values) {
    auto pointer = array->get_device_allocation().get_ptr();
    const void *data = values.data();
    std::size_t bytes = sizeof(values);
    if (device().upload_data(&pointer, &data, &bytes) != RhiResult::success)
      throw std::runtime_error("fixture upload failed");
  }
  TiNamedArgument array(int lane,
                        const char *name,
                        std::initializer_list<uint32_t> shape,
                        TiDataType type,
                        const void *initial = nullptr) {
    TiNamedArgument argument{};
    argument.name = name;
    argument.argument.type = TI_ARGUMENT_TYPE_NDARRAY;
    auto &value = argument.argument.value.ndarray;
    value.elem_type = type;
    value.shape.dim_count = shape.size();
    std::copy(shape.begin(), shape.end(), value.shape.dims);
    uint64_t bytes = 4;
    for (auto extent : shape)
      bytes *= extent;
    const TiMemoryAllocateInfo options{bytes, TI_TRUE, TI_TRUE, TI_FALSE,
                                       TI_MEMORY_USAGE_STORAGE_BIT};
    value.memory = ti_allocate_memory(runtimes[lane], &options);
    require_success();
    void *mapped = ti_map_memory(runtimes[lane], value.memory);
    require_success();
    if (!mapped)
      throw std::runtime_error("fixture map failed");
    if (initial)
      std::memcpy(mapped, initial, bytes);
    else
      std::memset(mapped, 0, bytes);
    ti_unmap_memory(runtimes[lane], value.memory);
    require_success();
    return argument;
  }
  void initialize(bool writer) {
    writes = writer;
    const auto make = [&](DataType dtype, std::vector<int> shape) {
      return program.create_ndarray(dtype, shape, ExternalArrayLayout::kNull,
                                    false);
    };
    auto *vertices = make(PrimitiveType::f32, {3, 3});
    auto *indices = make(PrimitiveType::i32, {1, 3});
    auto *transforms = make(PrimitiveType::f32, {1, 12});
    auto *rays = make(PrimitiveType::f32, {1, 8});
    auto *hits = make(PrimitiveType::f32, {1, 4});
    auto *hit_ids = make(PrimitiveType::i32, {1, 4});
    upload(vertices, std::array<float, 9>{-1, -1, 1, 1, -1, 1, 0, 1, 1});
    upload(indices, std::array<int32_t, 3>{0, 1, 2});
    const std::array<float, 8> ray{0, 0, 0, .01f, 0, 0, 1, 10};
    upload(rays, ray);
    upload(transforms,
           std::array<float, 12>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0});
    const auto v = storage::describe_ndarray_storage(*vertices);
    const auto ix = storage::describe_ndarray_storage(*indices);
    const auto tr = storage::describe_ndarray_storage(*transforms);
    blas = program.create_vulkan_triangle_blas_resource(3, 1);
    program.execute_vulkan_ray_geometry(program.prepare_vulkan_ray_geometry(
        blas, true, *v.descriptor, &*ix.descriptor, 3, 1));
    blas_refit = program.prepare_vulkan_ray_geometry(blas, true, *v.descriptor,
                                                     nullptr, 3, 1);
    for (int lane = 0; lane < 2; ++lane) {
      scenes[lane] = program.create_vulkan_instance_tlas_resource({blas});
      VulkanRayInstanceInfo instance;
      instance.custom_index = 17 + lane;
      program.vulkan_instance_tlas_build(scenes[lane], {instance}, false);
      ordinary_refits[lane] = program.prepare_vulkan_tlas_transforms(
          scenes[lane], *tr.descriptor, 1);
      ordinary_queries[lane] = program.prepare_vulkan_ray_query(
          scenes[lane], true, rays, hits, 1, hit_ids);
    }
    program.synchronize();
    const std::string query =
        R"({"kind":"native","provider":"vulkan_ray","operation":"query","version":1,
      "parameters":{"ray_count":1},"bindings":{"scene":"scene","rays":"rays","hits":"hits","hit_indices":"ids"}})";
    const std::string refit =
        R"({"kind":"native","provider":"vulkan_ray","operation":"instance_refit","version":1,
      "parameters":{},"bindings":{"scene":"scene","transforms":"transforms"}})";
    const std::string plan = "{\"version\":26,\"operations\":[" +
                             (writer ? refit + "," : "") + query + "]}";
    for (int lane = 0; lane < 2; ++lane) {
      borrowers[lane] = AotRuntimeClient::create(program);
      runtimes[lane] = reinterpret_cast<TiRuntime>(borrowers[lane]->handle());
      modules[lane] = reinterpret_cast<TiAotModule>(
          borrowers[lane]->import_jit_kernels({}));
      const auto root = scenes[writer ? lane : 0];
      const auto imported = reinterpret_cast<TiNativeResource>(
          borrowers[lane]->import_acceleration_structure(&program, root,
                                                         false));
      // A distinct registry handle can still alias the peer's original TLAS.
      const auto conflicting =
          lane ? reinterpret_cast<TiNativeResource>(
                     borrowers[lane]->import_acceleration_structure(
                         &program, scenes[0], false))
               : nullptr;
      observed[lane] = program.acquire_vulkan_ray_native_resource(root);
      BorrowScope scope(*borrowers[lane]);
      const TiVulkanExecutionDomainCreateInfo options{1, 1, TI_FALSE};
      domains[lane] =
          ti_create_vulkan_execution_domain_ext(runtimes[lane], &options);
      require_success();
      TiNamedArgument scene{};
      scene.name = "scene";
      scene.argument.type = TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT;
      scene.argument.value.native_resource = imported;
      std::vector<TiNamedArgument> arguments{
          scene, array(lane, "rays", {1, 8}, TI_DATA_TYPE_F32, ray.data()),
          array(lane, "hits", {1, 4}, TI_DATA_TYPE_F32),
          array(lane, "ids", {1, 4}, TI_DATA_TYPE_I32)};
      outputs[lane] = arguments[2].argument.value.ndarray.memory;
      ids[lane] = arguments[3].argument.value.ndarray.memory;
      const std::array<float, 12> pose{1, 0, 0, 0, 0, 1,
                                       0, 0, 0, 0, 1, float(lane + 1)};
      if (writer)
        arguments.push_back(
            array(lane, "transforms", {1, 12}, TI_DATA_TYPE_F32, pose.data()));
      if (lane) {
        auto invalid = arguments;
        invalid[0].argument.value.native_resource = conflicting;
        if (!writer)
          invalid.push_back(array(lane, "transforms", {1, 12}, TI_DATA_TYPE_F32,
                                  pose.data()));
        const auto invalid_plan =
            "{\"version\":26,\"operations\":[" + refit + "," + query + "]}";
        const auto before =
            device().queue_submission_snapshot().queue_submit_calls;
        auto rejected = ti_prepare_execution_plan_on_vulkan_domain_ext(
            runtimes[lane], modules[lane], domains[lane], invalid_plan.size(),
            invalid_plan.data(), invalid.size(), invalid.data(), 0, nullptr, 0,
            nullptr);
        EXPECT_EQ(rejected, nullptr);
        char message[1024]{};
        uint64_t size = sizeof(message);
        EXPECT_NE(ti_get_last_error(&size, message), TI_ERROR_SUCCESS);
        EXPECT_NE(std::string(message).find("loan"), std::string::npos);
        ti_set_last_error(TI_ERROR_SUCCESS, "");
        EXPECT_EQ(device().queue_submission_snapshot().queue_submit_calls,
                  before);
        // The one-instance budget must remain available to the valid plan.
      }
      executions[lane] = ti_prepare_execution_plan_on_vulkan_domain_ext(
          runtimes[lane], modules[lane], domains[lane], plan.size(),
          plan.data(), arguments.size(), arguments.data(), 0, nullptr, 0,
          nullptr);
      require_success();
      if (!executions[lane])
        throw std::runtime_error("fixture prepare failed");
      ti_initialize_prepared_ext(runtimes[lane], executions[lane]);
      require_success();
    }
    program.retain_runtime_client(gates);
  }
  void submit_pending(bool block_default) {
    if (block_default)
      gates->gates[2] =
          std::make_unique<test_detail::TimelineGate>(device(), 0);
    for (int lane = 0; lane < 2; ++lane) {
      BorrowScope scope(*borrowers[lane]);
      TiVulkanExecutionDomainInfo info{};
      ti_get_vulkan_execution_domain_info_ext(domains[lane], &info);
      require_success();
      gates->gates[lane] = std::make_unique<test_detail::TimelineGate>(
          device(), info.queue_index);
      const auto before =
          device().queue_submission_snapshot().queue_submit_calls;
      EXPECT_TRUE(ti_try_submit_prepared_ext(runtimes[lane], executions[lane],
                                             0, nullptr, &completions[lane]));
      require_success();
      EXPECT_EQ(device().queue_submission_snapshot().queue_submit_calls,
                before + 1);
      TiCompletion busy = nullptr;
      EXPECT_FALSE(ti_try_submit_prepared_ext(runtimes[lane], executions[lane],
                                              0, nullptr, &busy));
      require_success();
      EXPECT_EQ(busy, nullptr);
      EXPECT_EQ(device().queue_submission_snapshot().queue_submit_calls,
                before + 1);
      EXPECT_EQ(ti_query_completion_ext(completions[lane]), TI_FALSE);
    }
  }
  void expect_output(int lane) {
    BorrowScope scope(*borrowers[lane]);
    auto &runtime = *reinterpret_cast<::Runtime *>(runtimes[lane]);
    for (bool integer : {false, true}) {
      const auto allocation =
          runtime.get_memory(integer ? ids[lane] : outputs[lane]).allocation;
      void *mapped = nullptr;
      ASSERT_EQ(device().map(allocation, &mapped), RhiResult::success);
      if (integer) {
        const auto *data = static_cast<const int *>(mapped);
        EXPECT_EQ(data[0], 0);
        EXPECT_EQ(data[1], 0);
        EXPECT_EQ(data[2], writes ? 17 + lane : 17);
        EXPECT_EQ(data[3], 1);
      } else {
        const auto *data = static_cast<const float *>(mapped);
        EXPECT_FLOAT_EQ(data[0], writes ? float(lane + 2) : 1.f);
        EXPECT_FLOAT_EQ(data[1], .25f);
        EXPECT_FLOAT_EQ(data[2], .5f);
      }
      device().unmap(allocation);
    }
  }
};
}  // namespace

void check_program_as_pending_retirement(bool writes) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  PendingProgramAs session;
  if (!session.available())
    GTEST_SKIP();
  session.initialize(writes);
  session.submit_pending(true);
  const auto submissions =
      session.device().queue_submission_snapshot().queue_submit_calls;
  // Packets prepared before either loan must still use the same admission
  // identities. Two TLAS writers may share the read-only BLAS, but ordinary
  // BLAS refit must not invalidate either outstanding traversal.
  EXPECT_ANY_THROW(
      session.program.execute_vulkan_ray_geometry(session.blas_refit));
  EXPECT_ANY_THROW(session.program.execute_vulkan_tlas_transforms(
      session.ordinary_refits[0]));
  if (writes)
    EXPECT_ANY_THROW(
        session.program.execute_vulkan_ray_query(session.ordinary_queries[0]));
  EXPECT_EQ(session.device().queue_submission_snapshot().queue_submit_calls,
            submissions);

  TiVulkanExecutionInstance instance = nullptr;
  {
    BorrowScope scope(*session.borrowers[0]);
    instance = ti_get_prepared_vulkan_instance_ext(session.runtimes[0],
                                                   session.executions[0]);
    require_success();
    ASSERT_NE(instance, nullptr);
  }
  test_detail::PendingAction close(session.gates, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, "");
    ti_retire_vulkan_execution_instance_ext(instance);
    require_success();
  });
  EXPECT_EQ(close.status(100ms), std::future_status::timeout);
  EXPECT_FALSE(session.observed[0].expired());
  session.gates->release(0);
  const auto closed = close.status(2s);
  EXPECT_EQ(closed, std::future_status::ready)
      << "instance retirement must not wait for the default queue or peer "
         "borrower";
  if (closed != std::future_status::ready)
    session.gates->release_all();
  EXPECT_NO_THROW(close.finish());
  {
    BorrowScope scope(*session.borrowers[0]);
    ti_finish_prepared_instance_retirement_ext(session.runtimes[0],
                                               session.executions[0]);
    require_success();
    session.executions[0] = nullptr;
    ti_release_vulkan_execution_instance_ext(instance);
  }
  EXPECT_EQ(ti_query_completion_ext(session.completions[0]), TI_TRUE);
  EXPECT_EQ(ti_query_completion_ext(session.completions[1]), TI_FALSE);
  session.expect_output(0);
  // Only inspect admission here: executing on the deliberately blocked default
  // queue would obscure which prefix retirement actually waited for.
  if (writes) {
    EXPECT_NO_THROW(session.ordinary_queries[0].access->submit([] {}));
    EXPECT_ANY_THROW(session.ordinary_queries[1].access->submit([] {}));
  } else {
    EXPECT_NO_THROW(session.ordinary_queries[0].access->submit([] {}));
    EXPECT_ANY_THROW(session.ordinary_refits[0].access->submit([] {}));
  }
  EXPECT_ANY_THROW(session.blas_refit.access->submit([] {}));
  session.gates->release(1);
  ti_wait_completion_ext(session.completions[1]);
  require_success();
  session.expect_output(1);
  ti_close_vulkan_execution_domain_ext(session.domains[1]);
  require_success();
  session.gates->release(2);
  // Exercise the real old packets again after returning all conflicting loans.
  EXPECT_NO_THROW(
      session.program.execute_vulkan_ray_geometry(session.blas_refit));
  for (int lane = 0; lane < 2; ++lane) {
    EXPECT_NO_THROW(session.program.execute_vulkan_tlas_transforms(
        session.ordinary_refits[lane]));
    EXPECT_NO_THROW(session.program.execute_vulkan_ray_query(
        session.ordinary_queries[lane]));
  }
  session.program.synchronize();
}

TEST(VulkanProgramASImport, PendingReaderRetiresWithoutDefaultOrPeerWait) {
  check_program_as_pending_retirement(false);
}

TEST(VulkanProgramASImport, PendingTlasWriterReturnsOnlyItsRoot) {
  check_program_as_pending_retirement(true);
}

void check_program_as_pending_reset(bool writes) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  PendingProgramAs session;
  if (!session.available())
    GTEST_SKIP();
  session.initialize(writes);
  session.submit_pending(false);
  for (int lane = 0; lane < 2; ++lane) {
    session.program.destroy_vulkan_ray_resource(session.scenes[lane]);
    session.scenes[lane] = 0;
    BorrowScope scope(*session.borrowers[lane]);
    // Host handle release is not the GPU retirement boundary.
    ti_destroy_prepared_ext(session.runtimes[lane], session.executions[lane]);
    session.executions[lane] = nullptr;
    require_success();
  }
  session.program.destroy_vulkan_ray_resource(session.blas);
  session.blas = 0;
  const auto owner = session.program.runtime_fault_domain();
  test_detail::PendingAction reset(session.gates,
                                   [&] { session.program.finalize(); });
  EXPECT_EQ(reset.status(100ms), std::future_status::timeout);
  for (const auto &root : session.observed)
    EXPECT_FALSE(root.expired());
  session.gates->release(0);
  EXPECT_EQ(reset.status(100ms), std::future_status::timeout);
  // No backend completion query while Program is finalizing.
  for (int lane = 0; lane < 2; ++lane) {
    TiVulkanExecutionDomainInfo info{};
    ti_get_vulkan_execution_domain_info_ext(session.domains[lane], &info);
    require_success();
    EXPECT_FALSE(info.closed);
    EXPECT_FALSE(session.observed[lane].expired());
  }
  session.gates->release(1);
  EXPECT_EQ(reset.status(2s), std::future_status::ready);
  EXPECT_NO_THROW(reset.finish());
  EXPECT_EQ(owner->state(), RuntimeLifecycleState::kFinalized);
  for (int lane = 0; lane < 2; ++lane) {
    EXPECT_TRUE(session.borrowers[lane]->closed());
    EXPECT_TRUE(session.observed[lane].expired());
    EXPECT_EQ(ti_query_completion_ext(session.completions[lane]), TI_TRUE);
    ti_wait_completion_ext(session.completions[lane]);
    require_success();
  }
}

TEST(VulkanProgramASImport, ResetDrainsBothReadersAfterSourceDeletion) {
  check_program_as_pending_reset(false);
}

TEST(VulkanProgramASImport, ResetDrainsBothWritersAfterSourceDeletion) {
  check_program_as_pending_reset(true);
}

TEST(VulkanProgramASImport, OrderedAliasesRollbackAndOwnerRetirement) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  auto program = std::make_unique<Program>(Arch::vulkan);
  program->materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program->get_compute_device());
  if (!device.vk_caps().acceleration_structure || !device.vk_caps().ray_query)
    GTEST_SKIP();
  auto first = AotRuntimeClient::create(*program);
  auto second = AotRuntimeClient::create(*program);
  auto &registry = reinterpret_cast<::Runtime *>(first->handle())->native_resource_registry();
  auto &peer = reinterpret_cast<::Runtime *>(second->handle())->native_resource_registry();
  const auto a = program->create_vulkan_triangle_blas_resource(3, 1);
  const auto b = program->create_vulkan_triangle_blas_resource(3, 1);
  const auto tlas = program->create_vulkan_instance_tlas_resource({a, a, b});
  auto source = program->acquire_vulkan_ray_native_resource(tlas);
  auto original_a = program->acquire_vulkan_ray_native_resource(a);
  auto original_b = program->acquire_vulkan_ray_native_resource(b);
  auto instances = std::dynamic_pointer_cast<native_vulkan_ray::InstanceAccess>(source);
  ASSERT_EQ(instances->instance_blases().size(), 3);
  EXPECT_EQ(instances->dependencies().size(), 2);
  auto import = [&](AotRuntimeClient &client) {
    return reinterpret_cast<TiNativeResource>(
        client.import_acceleration_structure(program.get(), tlas, false));
  };
  EXPECT_THROW(first->import_acceleration_structure(nullptr, tlas, false),
               std::invalid_argument);
  const auto one = import(*first);
  const auto two = import(*first);
  const auto other = import(*second);
  EXPECT_NE(one, two);
  EXPECT_EQ(registry.get(one), source);
  EXPECT_EQ(registry.get(two), source);
  EXPECT_EQ(peer.get(other), source);
  const auto a0 = registry.resolve(one, {"blas_0"});
  const auto a1 = registry.resolve(one, {"blas_1"});
  const auto b2 = registry.resolve(one, {"blas_2"});
  EXPECT_EQ(a0, a1);
  EXPECT_EQ(registry.get(a0), original_a);
  EXPECT_EQ(registry.get(b2), original_b);
  EXPECT_NE(a0, registry.resolve(two, {"blas_0"}));
  EXPECT_EQ(registry.get(a0), registry.get(registry.resolve(two, {"blas_1"})));
  EXPECT_ANY_THROW(registry.resolve(one, {"blas_3"}));

  // Failing after publishing an input must release it without a queue submit.
  const auto rollback = program->create_vulkan_triangle_blas_resource(3, 1);
  auto rollback_owner = program->acquire_vulkan_ray_native_resource(rollback);
  std::weak_ptr<NativeResource> weak_rollback = rollback_owner;
  const auto submissions = device.queue_submission_snapshot().queue_submit_calls;
  EXPECT_ANY_THROW(registry.publish_borrowed(nullptr, {{"blas_0", rollback_owner}}));
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, submissions);
  program->destroy_vulkan_ray_resource(rollback);
  rollback_owner.reset();
  program->synchronize();
  EXPECT_TRUE(weak_rollback.expired());

  program->destroy_vulkan_ray_resource(a);
  program->destroy_vulkan_ray_resource(b);
  const auto after_blas_close = import(*first);
  EXPECT_EQ(registry.get(registry.resolve(after_blas_close, {"blas_2"})), original_b);
  program->destroy_vulkan_ray_resource(tlas);
  EXPECT_ANY_THROW(import(*first));
  EXPECT_EQ(registry.get(one), source);
  registry.destroy(one);
  EXPECT_ANY_THROW(registry.get(one));
  EXPECT_ANY_THROW(registry.get(a0));
  EXPECT_ANY_THROW(registry.get(b2));
  EXPECT_EQ(registry.get(two), source);
  first->close();
  EXPECT_EQ(peer.get(other), source);
  std::weak_ptr<NativeResource> weak_scene = source, weak_a = original_a, weak_b = original_b;
  source.reset();
  instances.reset();
  original_a.reset();
  original_b.reset();
  program.reset();
  EXPECT_TRUE(second->closed());
  EXPECT_TRUE(weak_scene.expired());
  EXPECT_TRUE(weak_a.expired());
  EXPECT_TRUE(weak_b.expired());
  EXPECT_NO_THROW(second->close());
}

void check_live_output_bootstrap(bool view) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  PendingProgramAs session;
  if (session.device().created_compute_queue_count() < 2)
    GTEST_SKIP();
  session.borrowers[0] = AotRuntimeClient::create(session.program);
  session.runtimes[0] =
      reinterpret_cast<TiRuntime>(session.borrowers[0]->handle());
  const uint32_t value = 42;
  const std::string bytes(reinterpret_cast<const char *>(&value),
                          sizeof(value));
  const auto key = picosha2::hash256_hex_string(bytes);
  session.modules[0] = reinterpret_cast<TiAotModule>(
      session.borrowers[0]->import_jit_kernels({}, {{key, bytes}}));
  TiNamedArgument output{};
  {
    BorrowScope scope(*session.borrowers[0]);
    const TiVulkanExecutionDomainCreateInfo options{1, 1, TI_FALSE};
    session.domains[0] =
        ti_create_vulkan_execution_domain_ext(session.runtimes[0], &options);
    require_success();
    output = session.array(0, "output", {1}, TI_DATA_TYPE_U32);
  }
  std::string plan = R"({"version":26,"operations":[
    {"kind":"copy","source":"seed","destination":")" +
                     std::string(view ? "alias" : "output") +
                     R"("}],"resources":[
    {"kind":"ndarray","name":"seed","dtype":"u32","shape":[1],"initial_data":")" +
                     key + "\"}";
  if (view)
    plan +=
        R"(,{"kind":"ndarray_view","name":"alias","array":"output","offset":0,"shape":[1]})";
  plan += "]}";
  session.program.synchronize();
  session.program.retain_runtime_client(session.gates);
  session.gates->gates[0] =
      std::make_unique<test_detail::TimelineGate>(session.device(), 0);
  test_detail::PendingAction prepare(session.gates, [&] {
    BorrowScope scope(*session.borrowers[0]);
    session.executions[0] = ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtimes[0], session.modules[0], session.domains[0],
        plan.size(), plan.data(), 1, &output, 0, nullptr, 0, nullptr);
    require_success();
    if (!session.executions[0])
      throw std::runtime_error("Cold payload preparation failed");
  });
  const auto status = prepare.status(2s);
  EXPECT_EQ(status, std::future_status::ready);
  if (status != std::future_status::ready)
    session.gates->release(0);
  prepare.finish();
  // The private seed no longer needs the producer queue, but the public output
  // may have prior reads/writes. Its actual producer prefix must still apply.
  test_detail::PendingAction initialize(session.gates, [&] {
    BorrowScope scope(*session.borrowers[0]);
    ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
    require_success();
  });
  EXPECT_EQ(initialize.status(100ms), std::future_status::timeout);
  session.gates->release(0);
  initialize.finish();
  {
    BorrowScope scope(*session.borrowers[0]);
    ASSERT_TRUE(ti_try_submit_prepared_ext(session.runtimes[0],
                                           session.executions[0], 0, nullptr,
                                           &session.completions[0]));
    require_success();
  }
  ti_wait_completion_ext(session.completions[0]);
  require_success();
  auto &runtime = *reinterpret_cast<::Runtime *>(session.runtimes[0]);
  const auto allocation =
      runtime.get_memory(output.argument.value.ndarray.memory).allocation;
  void *mapped = nullptr;
  ASSERT_EQ(session.device().map(allocation, &mapped), RhiResult::success);
  EXPECT_EQ(*static_cast<uint32_t *>(mapped), value);
  session.device().unmap(allocation);
}
TEST(VulkanColdPreparation, PayloadUploadPreservesLiveOutputBootstrap) {
  check_live_output_bootstrap(false);
}
TEST(VulkanColdPreparation, PrivateViewHandlePreservesPublicParentBootstrap) {
  check_live_output_bootstrap(true);
}
void check_spd_preparation_prefix(bool private_pixels) {
  const auto *shader_path = std::getenv("TI_TEST_SPD_SHADER");
  if (!shader_path || !vulkan::is_vulkan_api_available())
    GTEST_SKIP() << "Provide the 129x131, seven-level mean SPD shader";
  std::ifstream input(shader_path, std::ios::binary);
  ASSERT_TRUE(input.good());
  const std::string shader((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  ASSERT_GE(shader.size(), 20);
  PendingProgramAs session;
  if (session.device().created_compute_queue_count() < 2)
    GTEST_SKIP();
  session.borrowers[0] = AotRuntimeClient::create(session.program);
  session.runtimes[0] =
      reinterpret_cast<TiRuntime>(session.borrowers[0]->handle());
  const auto key = picosha2::hash256_hex_string(shader);
  session.modules[0] = reinterpret_cast<TiAotModule>(
      session.borrowers[0]->import_jit_kernels({}, {{key, shader}}));
  TiNamedArgument pixels{};
  {
    BorrowScope scope(*session.borrowers[0]);
    const TiVulkanExecutionDomainCreateInfo options{1, 1, TI_FALSE};
    session.domains[0] =
        ti_create_vulkan_execution_domain_ext(session.runtimes[0], &options);
    require_success();
    if (!private_pixels)
      pixels = session.array(0, "pixels", {129, 131}, TI_DATA_TYPE_F32);
  }
  std::string plan = R"({"version":26,"resources":[)";
  if (private_pixels)
    plan +=
        R"({"kind":"ndarray","name":"pixels","dtype":"f32","shape":[129,131]},)";
  plan += R"(
    {"kind":"host_blob","name":"shader","data":")" +
          key + "\",\"byte_size\":" + std::to_string(shader.size()) + R"(},
    {"kind":"native","name":"source","provider":"image","operation":"sampled","version":1,
     "parameters":{"shape":[129,131],"format":"r32f","storage":true},"bindings":{}},
    {"kind":"native","name":"pyramid","provider":"image","operation":"sampled","version":1,
     "parameters":{"shape":[64,65],"mip_levels":7,"format":"r32f","storage":true},"bindings":{}},
    {"kind":"native","name":"plan","provider":"spd","operation":"plan","version":1,
     "parameters":{"reduction":"mean"},"bindings":{"source":"source","output":"pyramid","shader":"shader"}}],
    "initialization":[
    {"kind":"native","provider":"image","operation":"upload","version":1,
     "parameters":{},"bindings":{"image":"source","pixels":"pixels"}}],
    "operations":[
    {"kind":"native","provider":"spd","operation":"run","version":1,
     "parameters":{},"bindings":{"plan":"plan","source":"source","output":"pyramid"}}]})";
  session.program.synchronize();
  session.program.retain_runtime_client(session.gates);
  session.gates->gates[0] =
      std::make_unique<test_detail::TimelineGate>(session.device(), 0);
  test_detail::PendingAction prepare(session.gates, [&] {
    BorrowScope scope(*session.borrowers[0]);
    session.executions[0] = ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtimes[0], session.modules[0], session.domains[0],
        plan.size(), plan.data(), private_pixels ? 0 : 1,
        private_pixels ? nullptr : &pixels, 0, nullptr, 0, nullptr);
    require_success();
    if (!session.executions[0])
      throw std::runtime_error("SPD cold preparation failed");
    if (private_pixels) {
      ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
      require_success();
      if (!ti_try_submit_prepared_ext(session.runtimes[0],
                                      session.executions[0], 0, nullptr,
                                      &session.completions[0]))
        throw std::runtime_error("Private SPD submission was not admitted");
      require_success();
      ti_wait_completion_ext(session.completions[0]);
      require_success();
    }
  });
  const auto status = prepare.status(2s);
  EXPECT_EQ(status, std::future_status::ready);
  if (status != std::future_status::ready)
    session.gates->release(0);
  prepare.finish();
  if (private_pixels) {
    session.gates->release(0);
    return;
  }
  // The public pixel input still requires its real producer boundary. Only
  // the private counter's clear moved into the domain initialization phase.
  session.gates->release(0);
  {
    BorrowScope scope(*session.borrowers[0]);
    ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
    require_success();
    ASSERT_TRUE(ti_try_submit_prepared_ext(session.runtimes[0],
                                           session.executions[0], 0, nullptr,
                                           &session.completions[0]));
    require_success();
  }
  ti_wait_completion_ext(session.completions[0]);
  require_success();
}
TEST(VulkanColdPreparation, SpdCounterClearDoesNotWaitForDefaultQueue) {
  check_spd_preparation_prefix(false);
}
TEST(VulkanColdPreparation, PrivateSpdInitializationAndRunIgnoreDefaultQueue) {
  check_spd_preparation_prefix(true);
}
void check_image_preparation_prefix(bool private_pixels,
                                    bool unknown_factory = false) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  PendingProgramAs session;
  if (session.device().created_compute_queue_count() < 3)
    GTEST_SKIP();
  session.borrowers[0] = AotRuntimeClient::create(session.program);
  session.runtimes[0] =
      reinterpret_cast<TiRuntime>(session.borrowers[0]->handle());
  const std::array<float, 6> values{99, 3, -2, 7, 11, -99};
  const std::string bytes(reinterpret_cast<const char *>(values.data()),
                          sizeof(values));
  const auto key = picosha2::hash256_hex_string(bytes);
  session.modules[0] = reinterpret_cast<TiAotModule>(
      session.borrowers[0]->import_jit_kernels({}, {{key, bytes}}));
  TiNamedArgument pixels{};
  {
    BorrowScope scope(*session.borrowers[0]);
    const TiVulkanExecutionDomainCreateInfo options{1, 1, TI_FALSE};
    session.domains[0] =
        ti_create_vulkan_execution_domain_ext(session.runtimes[0], &options);
    require_success();
    if (!private_pixels)
      pixels =
          session.array(0, "pixels", {4}, TI_DATA_TYPE_F32, values.data() + 1);
  }
  std::string plan = R"({"version":26,"resources":[)";
  if (private_pixels)
    plan +=
        R"({"kind":"ndarray","name":"seed","dtype":"f32","shape":[6],"initial_data":")" +
        key +
        R"("},{"kind":"ndarray_view","name":"pixels","array":"seed","offset":1,"shape":[4]},)";
  plan +=
      R"({"kind":"native","name":"image","provider":"image","operation":"sampled","version":1,
    "parameters":{"shape":[4],"format":"r32f"},"bindings":{"pixels":"pixels"}}],
    "operations":[{"kind":"native","provider":"image","operation":"upload","version":1,
    "parameters":{},"bindings":{"image":"image","pixels":"pixels"}}]})";
  if (unknown_factory) {
    plan.resize(plan.size() - 2);
    plan += R"(,{"kind":"sort","keys":"pixels"}]})";
  }
  session.program.synchronize();
  session.program.retain_runtime_client(session.gates);
  session.gates->gates[0] =
      std::make_unique<test_detail::TimelineGate>(session.device(), 0);
  session.gates->gates[1] =
      std::make_unique<test_detail::TimelineGate>(session.device(), 2);
  test_detail::PendingAction prepare(session.gates, [&] {
    BorrowScope scope(*session.borrowers[0]);
    session.executions[0] = ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtimes[0], session.modules[0], session.domains[0],
        plan.size(), plan.data(), private_pixels ? 0 : 1,
        private_pixels ? nullptr : &pixels, 0, nullptr, 0, nullptr);
    require_success();
    if (!session.executions[0])
      throw std::runtime_error("Image preparation failed");
    if (private_pixels && !unknown_factory) {
      ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
      require_success();
      if (!ti_try_submit_prepared_ext(session.runtimes[0],
                                      session.executions[0], 0, nullptr,
                                      &session.completions[0]))
        throw std::runtime_error("Private image submission was not admitted");
      require_success();
      ti_wait_completion_ext(session.completions[0]);
      require_success();
    }
  });
  const auto status = prepare.status(private_pixels ? 2s : 100ms);
  EXPECT_EQ(status, private_pixels ? std::future_status::ready
                                   : std::future_status::timeout);
  if (unknown_factory && status == std::future_status::ready) {
    prepare.finish();
    test_detail::PendingAction initialize(session.gates, [&] {
      BorrowScope scope(*session.borrowers[0]);
      ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
      require_success();
    });
    EXPECT_EQ(initialize.status(100ms), std::future_status::timeout);
    session.gates->release(0);
    initialize.finish();
    session.gates->release(1);
    return;
  }
  // Public pixels still need their producer; the unrelated peer stays pending.
  session.gates->release(0);
  if (prepare.status(2s) != std::future_status::ready)
    session.gates->release(1);
  EXPECT_EQ(prepare.status(2s), std::future_status::ready);
  prepare.finish();
  session.gates->release(1);
  {
    BorrowScope scope(*session.borrowers[0]);
    ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
    require_success();
  }
}

TEST(VulkanColdPreparation, PrivateImagePixelsIgnoreDefaultAndPeer) {
  check_image_preparation_prefix(true);
}
TEST(VulkanColdPreparation, PublicImagePixelsStillWaitForProducer) {
  check_image_preparation_prefix(false);
}
TEST(VulkanColdPreparation, UndescribedNativeConstructionPreservesBootstrap) {
  check_image_preparation_prefix(true, true);
}

void check_kernel_preparation_prefix(bool field) {
  const auto *path = std::getenv("TI_TEST_PREPARATION_AOT_DIR");
  if (!path || !vulkan::is_vulkan_api_available())
    GTEST_SKIP() << "Requires the rootless and Field-dependent kernel fixture";
  PendingProgramAs session;
  if (session.device().created_compute_queue_count() < 3)
    GTEST_SKIP();
  session.borrowers[0] = AotRuntimeClient::create(session.program);
  session.runtimes[0] =
      reinterpret_cast<TiRuntime>(session.borrowers[0]->handle());
  {
    BorrowScope scope(*session.borrowers[0]);
    session.modules[0] = ti_load_aot_module(session.runtimes[0], path);
    require_success();
    ASSERT_NE(session.modules[0], nullptr);
    const TiVulkanExecutionDomainCreateInfo options{1, 1, TI_FALSE};
    session.domains[0] =
        ti_create_vulkan_execution_domain_ext(session.runtimes[0], &options);
    require_success();
  }
  const std::string plan =
      R"({"version":26,"resources":[{"kind":"ndarray","name":"out","dtype":"u32","shape":[1]}],"operations":[{"kind":"kernel","entry":")" +
      std::string(field ? "field" : "rootless") + R"(","arguments":["out"]}]})";
  session.program.synchronize();
  session.program.retain_runtime_client(session.gates);
  session.gates->gates[0] =
      std::make_unique<test_detail::TimelineGate>(session.device(), 0);
  session.gates->gates[1] =
      std::make_unique<test_detail::TimelineGate>(session.device(), 2);
  test_detail::PendingAction run(session.gates, [&] {
    BorrowScope scope(*session.borrowers[0]);
    session.executions[0] = ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtimes[0], session.modules[0], session.domains[0],
        plan.size(), plan.data(), 0, nullptr, 0, nullptr, 0, nullptr);
    require_success();
    ti_initialize_prepared_ext(session.runtimes[0], session.executions[0]);
    require_success();
    if (!ti_try_submit_prepared_ext(session.runtimes[0], session.executions[0],
                                    0, nullptr, &session.completions[0]))
      throw std::runtime_error("Kernel bootstrap fixture could not submit");
    require_success();
    ti_wait_completion_ext(session.completions[0]);
    require_success();
  });
  const auto status = run.status(field ? 100ms : 2s);
  EXPECT_EQ(status,
            field ? std::future_status::timeout : std::future_status::ready);
  session.gates->release(0);
  if (run.status(2s) != std::future_status::ready)
    session.gates->release(1);
  EXPECT_EQ(run.status(2s), std::future_status::ready);
  run.finish();
  session.gates->release(1);
}

TEST(VulkanColdPreparation, RootlessKernelIgnoresDefaultAndPeer) {
  check_kernel_preparation_prefix(false);
}
TEST(VulkanColdPreparation, FieldKernelPreservesProducerPrefix) {
  check_kernel_preparation_prefix(true);
}
}  // namespace taichi::lang
#endif
