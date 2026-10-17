#include "gtest/gtest.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

namespace taichi::lang {
namespace {

class MemoryAotDir : public io::VirtualDir {
 public:
  std::map<std::string, std::string> files;
  std::string short_read;
  mutable std::map<std::string, int> reads;

  bool get_file_size(const std::string &path, size_t &size) const override {
    auto it = files.find(path);
    if (it == files.end())
      return false;
    size = it->second.size();
    return true;
  }
  size_t load_file(const std::string &path,
                   void *dst,
                   size_t size) const override {
    ++reads[path];
    auto it = files.find(path);
    if (it == files.end())
      return 0;
    size = std::min(size, it->second.size());
    if (path == short_read && size)
      --size;
    std::memcpy(dst, it->second.data(), size);
    return size;
  }

  MemoryAotDir() {
    gfx::TaichiAotData data;
    data.metadata_version = gfx::TaichiAotData::kMetadataVersion;
    data.required_caps = {{"spirv_version", 0x10300}, {"spirv_has_int64", 1}};
    files["metadata.json"] = liong::json::print(liong::json::serialize(data));
    files["graphs.json"] = "[]";
  }

  void add_shader(const std::vector<uint32_t> &words) {
    gfx::TaichiAotData data;
    liong::json::deserialize(liong::json::parse(files["metadata.json"]), data);
    spirv::TaichiKernelAttributes kernel;
    kernel.name = "test";
    spirv::TaskAttributes task;
    task.name = "test_task";
    task.task_type = OffloadedTaskType::serial;
    kernel.tasks_attribs.push_back(task);
    data.kernels.push_back(kernel);
    data.kernel_metadata.emplace_back();
    files["metadata.json"] = liong::json::print(liong::json::serialize(data));
    files["test_task.spv"] = std::string(
        reinterpret_cast<const char *>(words.data()), words.size() * 4);
  }

  void add_alias(bool legacy) {
    gfx::TaichiAotData data;
    liong::json::deserialize(liong::json::parse(files["metadata.json"]), data);
    if (legacy) {
      data.metadata_version = 1;
      data.kernels.push_back(data.kernels.front());
      data.kernels.back().name = "alias";
      data.kernel_metadata.push_back(data.kernel_metadata.front());
    } else {
      data.kernel_aliases.push_back({"alias", 0});
    }
    files["metadata.json"] = liong::json::print(liong::json::serialize(data));
  }

  std::unique_ptr<aot::Module> load() {
    return gfx::make_aot_module(gfx::AotModuleParams(this, nullptr),
                                Arch::vulkan);
  }
};

TEST(GfxAotValidation, ExposesRequiredCapabilities) {
  MemoryAotDir dir;
  auto module = dir.load();
  ASSERT_FALSE(module->is_corrupted());
  EXPECT_EQ(module->get_required_caps().get(DeviceCapability::spirv_version),
            0x10300);
  EXPECT_EQ(module->get_required_caps().get(DeviceCapability::spirv_has_int64),
            1);
}

TEST(GfxAotValidation, RejectsMissingShortAndInvalidShaderReads) {
  const std::vector<uint32_t> header{0x07230203, 0x10300, 0, 1, 0};
  MemoryAotDir dir;
  dir.add_shader(header);
  ASSERT_FALSE(dir.load()->is_corrupted());
  dir.short_read = "test_task.spv";
  EXPECT_TRUE(dir.load()->is_corrupted());
  dir.short_read.clear();
  dir.files["test_task.spv"][0] = 0;
  EXPECT_TRUE(dir.load()->is_corrupted());
  dir.files["test_task.spv"].resize(4);
  EXPECT_TRUE(dir.load()->is_corrupted());
  dir.files.erase("test_task.spv");
  EXPECT_TRUE(dir.load()->is_corrupted());
}

TEST(GfxAotValidation, RejectsMalformedMetadata) {
  MemoryAotDir dir;
  dir.files["graphs.json"] = "{";
  EXPECT_TRUE(dir.load()->is_corrupted());
  dir.files["metadata.json"] = "{";
  EXPECT_TRUE(dir.load()->is_corrupted());
}

TEST(GfxAotValidation, OwnedSourcesDeferShaderReadsAndCacheFailures) {
  auto dir = std::make_shared<MemoryAotDir>();
  dir->add_shader({0});
  std::weak_ptr<MemoryAotDir> weak = dir;
  gfx::AotModuleParams params;
  params.source = dir;
  auto module = gfx::make_aot_module(params, Arch::vulkan);
  ASSERT_FALSE(module->is_corrupted());
  EXPECT_EQ(dir->reads["test_task.spv"], 0);
  dir.reset();
  params.source.reset();
  ASSERT_FALSE(weak.expired());
  EXPECT_EQ(module->get_kernel("absent"), nullptr);
  EXPECT_THROW(module->get_kernel("test"), aot::ArtifactLoadError);
  EXPECT_THROW(module->get_kernel("test"), aot::ArtifactLoadError);
  EXPECT_EQ(weak.lock()->reads["test_task.spv"], 1);
  module.reset();
  EXPECT_TRUE(weak.expired());
}

TEST(GfxAotValidation, AliasesShareDeferredFailuresAndLegacyEagerReads) {
  for (bool legacy : {false, true}) {
    auto dir = std::make_shared<MemoryAotDir>();
    dir->add_shader({0x07230203, 0x10300, 0, 1, 0});
    dir->add_alias(legacy);
    ASSERT_FALSE(dir->load()->is_corrupted());
    EXPECT_EQ(dir->reads["test_task.spv"], 1);
    dir->reads.clear();
    dir->files["test_task.spv"][0] = 0;
    gfx::AotModuleParams params;
    params.source = dir;
    auto module = gfx::make_aot_module(params, Arch::vulkan);
    ASSERT_FALSE(module->is_corrupted());
    EXPECT_EQ(dir->reads["test_task.spv"], 0);
    EXPECT_THROW(module->get_kernel("alias"), aot::ArtifactLoadError);
    EXPECT_THROW(module->get_kernel("test"), aot::ArtifactLoadError);
    EXPECT_EQ(dir->reads["test_task.spv"], 1);
  }
}

TEST(GfxAotValidation, RejectsInvalidAliases) {
  MemoryAotDir dir;
  dir.add_shader({0x07230203, 0x10300, 0, 1, 0});
  gfx::TaichiAotData data;
  liong::json::deserialize(liong::json::parse(dir.files["metadata.json"]), data);
  for (const gfx::AotKernelAlias &alias :
       std::vector<gfx::AotKernelAlias>{{"alias", 1}, {"test", 0}, {"", 0}}) {
    data.kernel_aliases = {alias};
    dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
    EXPECT_TRUE(dir.load()->is_corrupted());
  }
  data.kernel_aliases = {{"alias", 0}, {"alias", 0}};
  dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
  EXPECT_TRUE(dir.load()->is_corrupted());
  data.kernel_aliases = {{"alias", 0}};
  data.metadata_version = 1;
  dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
  EXPECT_TRUE(dir.load()->is_corrupted());
}

TEST(GfxAotValidation, RejectsUndeclaredRootBindings) {
  MemoryAotDir dir;
  dir.add_shader({0x07230203, 0x10300, 0, 1, 0});
  gfx::TaichiAotData data;
  liong::json::deserialize(liong::json::parse(dir.files["metadata.json"]), data);
  data.root_buffer_size = 4;
  data.root_buffer_sizes = {4};
  data.kernel_metadata[0] = {1, {0}};
  spirv::TaskAttributes::BufferBind bind;
  bind.buffer = {spirv::TaskAttributes::BufferType::Root, 0};
  data.kernels[0].tasks_attribs[0].buffer_binds = {bind};
  dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
  EXPECT_FALSE(dir.load()->is_corrupted());
  data.kernel_metadata[0].used_snode_tree_ids.clear();
  dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
  EXPECT_TRUE(dir.load()->is_corrupted());
  data.kernels[0].tasks_attribs[0].buffer_binds[0].buffer.root_id.clear();
  dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
  EXPECT_TRUE(dir.load()->is_corrupted());
}

TEST(GfxAotValidation, StorageImageArgumentsRequireQualifiedMetadata) {
  MemoryAotDir dir;
  dir.add_shader({0x07230203, 0x10300, 0, 1, 0});
  gfx::TaichiAotData data;
  liong::json::deserialize(liong::json::parse(dir.files["metadata.json"]), data);
  spirv::KernelContextAttributes::ArgAttributes argument;
  argument.ptype = ParameterType::kRWTexture;
  argument.format = BufferFormat::r32u;
  argument.field_dim = 2;
  argument.texture_lod = 2;
  auto update = [&] {
    auto json = liong::json::serialize(data.kernels[0].ctx_attribs);
    json["arg_attribs_vec_"] = liong::json::serialize(
        std::vector<std::pair<std::vector<int>,
                              spirv::KernelContextAttributes::ArgAttributes>>{
            {{0}, argument}});
    liong::json::deserialize(json, data.kernels[0].ctx_attribs);
    dir.files["metadata.json"] = liong::json::print(liong::json::serialize(data));
    return dir.load()->is_corrupted();
  };
  EXPECT_FALSE(update());
  data.metadata_version = 2;
  EXPECT_TRUE(update());
  data.metadata_version = 4;
  argument.format = BufferFormat::unknown;
  EXPECT_TRUE(update());
  argument.format = BufferFormat::r32u;
  argument.texture_lod = -1;
  EXPECT_TRUE(update());
  argument.texture_lod = 2;
  argument.field_dim = 3;
  EXPECT_TRUE(update());
}

}  // namespace
}  // namespace taichi::lang
