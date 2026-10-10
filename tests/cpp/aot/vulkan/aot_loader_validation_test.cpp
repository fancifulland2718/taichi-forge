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

}  // namespace
}  // namespace taichi::lang
