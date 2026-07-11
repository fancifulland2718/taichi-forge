#include "program_impl.h"

namespace taichi::lang {

ProgramImpl::ProgramImpl(CompileConfig &config_) : config(&config_) {
}

void ProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  // FIXME: Eventually all the backends should implement this
  TI_NOT_IMPLEMENTED;
}

void ProgramImpl::dump_cache_data_to_disk() {
  if (!config->offline_cache && !kernel_com_mgr_) {
    return;
  }
  auto &mgr =
      kernel_com_mgr_ ? *kernel_com_mgr_ : get_kernel_compilation_manager();
  if (!config->offline_cache) {
    mgr.clear();
    return;
  }
  mgr.dump();
  mgr.clean_offline_cache(offline_cache::string_to_clean_cache_policy(
                              config->offline_cache_cleaning_policy),
                          config->offline_cache_max_size_of_files,
                          config->offline_cache_cleaning_factor, config->arch);
}

KernelCompilationManager &ProgramImpl::get_kernel_compilation_manager() {
  std::call_once(kernel_com_mgr_once_, [this] {
    KernelCompilationManager::Config cfg;
    cfg.offline_cache_path = config->offline_cache_file_path;
    cfg.kernel_compiler = make_kernel_compiler();
    kernel_com_mgr_ =
        std::make_unique<KernelCompilationManager>(std::move(cfg));
  });
  TI_ASSERT(kernel_com_mgr_ != nullptr);
  return *kernel_com_mgr_;
}

KernelLauncher &ProgramImpl::get_kernel_launcher() {
  std::call_once(kernel_launcher_once_,
                 [this] { kernel_launcher_ = make_kernel_launcher(); });
  TI_ASSERT(kernel_launcher_ != nullptr);
  return *kernel_launcher_;
}

}  // namespace taichi::lang
