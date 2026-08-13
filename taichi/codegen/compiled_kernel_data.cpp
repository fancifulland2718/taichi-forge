#include "compiled_kernel_data.h"
#include "taichi/common/logging.h"

#include "picosha2.h"

namespace taichi::lang {

void qualify_dense_snode_relocation(SNodeRelocationDescriptor &descriptor) {
  if (!descriptor.compiler_emitted ||
      !descriptor.has_snode_tree_dependencies) {
    return;
  }
  if (!arch_is_cpu(descriptor.backend) && descriptor.backend != Arch::cuda &&
      descriptor.backend != Arch::vulkan) {
    return;
  }
  const bool has_sparse_state = std::find(
      descriptor.blockers.begin(), descriptor.blockers.end(),
      SNodeRelocationBlocker::sparse_state_not_qualified) !=
      descriptor.blockers.end();
  if (has_sparse_state) {
    return;
  }

  descriptor.compiler_embedded_state_fully_classified = true;
  descriptor.reuse_admitted = true;
  descriptor.relocation_class = SNodeRelocationClass::partially_relocatable;
  descriptor.blockers.clear();
  for (auto &task : descriptor.tasks) {
    task.relocation_class = SNodeRelocationClass::partially_relocatable;
    task.generation_bound_state = {
        SNodeRelocationState::tree_identity,
        SNodeRelocationState::root_allocation,
        SNodeRelocationState::runtime_state,
        SNodeRelocationState::backend_registration,
    };
  }
}

void CompiledKernelData::initialize_generation_bound_snode_relocation_descriptor(
    bool compiler_emitted,
    SNodeRelocationStructure structures) {
  SNodeRelocationDescriptor descriptor;
  descriptor.backend = arch();
  descriptor.compiler_emitted = compiler_emitted;
  descriptor.has_snode_tree_dependencies = has_snode_tree_dependencies();
  const auto manifests = task_manifest();
  if (!descriptor.has_snode_tree_dependencies) {
    descriptor.compiler_embedded_state_fully_classified = true;
    descriptor.relocation_class = SNodeRelocationClass::not_applicable;
    descriptor.tasks.reserve(manifests.size());
    for (const auto &manifest : manifests) {
      SNodeTaskRelocationDescriptor task;
      task.task_index = manifest.task_index;
      task.task_type = manifest.task_type;
      task.relocation_class = SNodeRelocationClass::not_applicable;
      descriptor.tasks.push_back(std::move(task));
    }
    set_snode_relocation_descriptor(std::move(descriptor));
    return;
  }

  descriptor.compiler_embedded_state_fully_classified = false;
  descriptor.reuse_admitted = false;
  descriptor.relocation_class = SNodeRelocationClass::generation_bound;
  descriptor.blockers = {
      SNodeRelocationBlocker::compiler_embedded_state_unclassified,
      SNodeRelocationBlocker::executable_and_generation_binding_not_separated,
      SNodeRelocationBlocker::in_flight_rebind_not_qualified,
      SNodeRelocationBlocker::graph_masked_rebind_not_qualified,
  };
  if (arch_uses_llvm(descriptor.backend)) {
    descriptor.blockers.push_back(
        SNodeRelocationBlocker::llvm_registration_generation_specific);
  } else if (arch_uses_spirv(descriptor.backend)) {
    descriptor.blockers.push_back(
        SNodeRelocationBlocker::spirv_registration_generation_specific);
  }

  descriptor.tasks.reserve(manifests.size());
  const bool has_sparse_structure =
      structures != SNodeRelocationStructure::none;
  bool has_sparse_task = has_sparse_structure;
  for (const auto &manifest : manifests) {
    SNodeTaskRelocationDescriptor task;
    task.task_index = manifest.task_index;
    task.task_type = manifest.task_type;
    task.relocation_class = SNodeRelocationClass::generation_bound;
    task.generation_bound_state = {
        SNodeRelocationState::tree_identity,
        SNodeRelocationState::root_allocation,
        SNodeRelocationState::runtime_state,
        SNodeRelocationState::backend_registration,
        SNodeRelocationState::compiler_embedded_state_unclassified,
    };
    if (manifest.task_type == OffloadedTaskType::listgen) {
      has_sparse_task = true;
      task.generation_bound_state.push_back(
          SNodeRelocationState::sparse_listgen_state);
      task.generation_bound_state.push_back(
          SNodeRelocationState::sparse_active_list_metadata);
    } else if (manifest.task_type == OffloadedTaskType::gc ||
               (manifest.task_type == OffloadedTaskType::struct_for &&
                has_sparse_structure)) {
      has_sparse_task = true;
      task.generation_bound_state.push_back(
          SNodeRelocationState::sparse_allocator_state);
    }
    if (has_snode_relocation_structure(
            structures, SNodeRelocationStructure::pointer)) {
      task.generation_bound_state.push_back(
          SNodeRelocationState::pointer_allocator_and_list_state);
    }
    if (has_snode_relocation_structure(
            structures, SNodeRelocationStructure::bitmasked)) {
      task.generation_bound_state.push_back(
          SNodeRelocationState::bitmasked_activity_state);
    }
    if (has_snode_relocation_structure(
            structures, SNodeRelocationStructure::dynamic)) {
      task.generation_bound_state.push_back(
          SNodeRelocationState::dynamic_chunk_and_length_state);
    }
    if (has_snode_relocation_structure(structures,
                                       SNodeRelocationStructure::hash)) {
      task.generation_bound_state.push_back(
          SNodeRelocationState::hash_bucket_tombstone_and_pool_state);
    }
    descriptor.tasks.push_back(std::move(task));
  }
  if (has_sparse_task) {
    descriptor.blockers.push_back(
        SNodeRelocationBlocker::sparse_state_not_qualified);
  }
  qualify_dense_snode_relocation(descriptor);
  set_snode_relocation_descriptor(std::move(descriptor));
}

std::string CompiledKernelData::make_task_identity(
    std::size_t task_index,
    OffloadedTaskType task_type) const {
  if (kernel_identity_.empty()) {
    return {};
  }
  return fmt::format("tf:{}:{}:{}", kernel_identity_, task_index,
                     offloaded_task_type_name(task_type));
}

static CompiledKernelData::Err translate_err(CompiledKernelDataFile::Err err) {
  switch (err) {
    case CompiledKernelDataFile::Err::kNoError:
      return CompiledKernelData::Err::kNoError;
    case CompiledKernelDataFile::Err::kNotTicFile:
      return CompiledKernelData::Err::kNotTicFile;
    case CompiledKernelDataFile::Err::kCorruptedFile:
      return CompiledKernelData::Err::kCorruptedFile;
    case CompiledKernelDataFile::Err::kOutOfMemory:
      return CompiledKernelData::Err::kOutOfMemory;
    case CompiledKernelDataFile::Err::kIOStreamError:
      return CompiledKernelData::Err::kIOStreamError;
  }
  return CompiledKernelData::Err::kUnknown;
}

CompiledKernelDataFile::Err CompiledKernelDataFile::dump(std::ostream &os) {
  try {
    update_hash();
    std::uint32_t arch = static_cast<std::uint32_t>(arch_);
    std::uint64_t metadata_size = metadata_.size();
    std::uint64_t src_code_size = src_code_.size();
    bool io_success =
        os.write(head_, std::size(head_)) &&
        os.write((const char *)&arch, sizeof(arch)) &&
        os.write((const char *)&metadata_size, sizeof(metadata_size)) &&
        os.write((const char *)&src_code_size, sizeof(src_code_size)) &&
        os.write((const char *)metadata_.data(), metadata_size) &&
        os.write((const char *)src_code_.data(), src_code_size) &&
        os.write((const char *)hash_.data(), kHashSize);
    if (!io_success) {
      return Err::kIOStreamError;
    }
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemory;
  }
  return Err::kNoError;
}

CompiledKernelDataFile::Err CompiledKernelDataFile::load(std::istream &is) {
  try {
    if (!is.read(head_, std::size(head_))) {
      return Err::kIOStreamError;
    } else if (std::strncmp(head_, kHeadStr, kHeadSize) != 0) {
      return Err::kNotTicFile;
    }
    std::uint32_t arch;
    std::uint64_t metadata_size;
    std::uint64_t src_code_size;
    bool io_success = is.read((char *)&arch, sizeof(arch)) &&
                      is.read((char *)&metadata_size, sizeof(metadata_size)) &&
                      is.read((char *)&src_code_size, sizeof(src_code_size));
    if (!io_success) {
      return Err::kIOStreamError;
    }
    arch_ = static_cast<Arch>(arch);
    metadata_.resize(metadata_size);
    src_code_.resize(src_code_size);
    hash_.resize(kHashSize);
    io_success = is.read((char *)metadata_.data(), metadata_size) &&
                 is.read((char *)src_code_.data(), src_code_size) &&
                 is.read((char *)hash_.data(), kHashSize);
    if (!io_success) {
      return Err::kIOStreamError;
    }
    if (update_hash()) {
      return Err::kCorruptedFile;
    }
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemory;
  }
  return Err::kNoError;
}

bool CompiledKernelDataFile::update_hash() {
  picosha2::hash256_one_by_one hasher;
  hasher.process(metadata_.begin(), metadata_.end());
  hasher.process(src_code_.begin(), src_code_.end());
  hasher.finish();
  auto hash = picosha2::get_hash_hex_string(hasher);
  if (hash == hash_) {
    return false;
  }
  hash_ = std::move(hash);
  TI_ASSERT(hash_.size() == kHashSize);
  return true;
}

#if !defined(TI_WITH_LLVM)
CompiledKernelData::Creator *const CompiledKernelData::llvm_creator = nullptr;
#endif

#if !defined(TI_WITH_VULKAN) && !defined(TI_WITH_OPENGL) && \
    !defined(TI_WITH_DX11) && !defined(TI_WITH_METAL)
CompiledKernelData::Creator *const CompiledKernelData::spriv_creator = nullptr;
#endif

CompiledKernelData::Err CompiledKernelData::load(std::istream &is) {
  try {
    Err err = Err::kNoError;
    CompiledKernelDataFile file;
    if (err = translate_err(file.load(is)); err != Err::kNoError) {
      return err;
    }
    const auto result = load_impl(file);
    if (result == Err::kNoError) {
      initialize_generation_bound_snode_relocation_descriptor(false);
    }
    return result;
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemory;
  }
}

CompiledKernelData::Err CompiledKernelData::dump(std::ostream &os) const {
  try {
    Err err = Err::kNoError;
    CompiledKernelDataFile file;
    if (err = dump_impl(file); err != Err::kNoError) {
      return err;
    }
    return translate_err(file.dump(os));
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemory;
  }
}

// static functions
std::unique_ptr<CompiledKernelData> CompiledKernelData::load(std::istream &is,
                                                             Err *p_err) {
  Err err = Err::kNoError;
  CompiledKernelDataFile file;
  std::unique_ptr<CompiledKernelData> result{nullptr};
  try {
    err = translate_err(file.load(is));
    if (err == Err::kNoError) {
      result = create(file.arch(), err);
    }
    if (err == Err::kNoError) {
      TI_ASSERT(result);
      err = result->load_impl(file);
      if (err == Err::kNoError) {
        result->initialize_generation_bound_snode_relocation_descriptor(false);
      }
    }
    if (err != Err::kNoError) {
      result = nullptr;
    }
  } catch (std::bad_alloc &) {
    err = Err::kOutOfMemory;
  }
  if (p_err) {
    *p_err = err;
  }
  return result;
}

std::string CompiledKernelData::get_err_msg(Err err) {
  switch (err) {
    case Err::kNoError:
      return "Success";
    case Err::kNotTicFile:
      return "The file is not TIC file";
    case Err::kCorruptedFile:
      return "The file was corrupted";
    case Err::kParseMetadataFailed:
      return "Parse metadata failed";
    case Err::kParseSrcCodeFailed:
      return "Parse src code failed";
    case Err::kArchNotMatched:
      return "Arch not matched";
    case Err::kSerMetadataFailed:
      return "Serialize metadata failed";
    case Err::kSerSrcCodeFailed:
      return "Serialize src code failed";
    case Err::kIOStreamError:
      return "IO error";
    case Err::kOutOfMemory:
      return "Out of memory";
    case Err::kTiWithoutLLVM:
      return "The taichi is not built with llvm";
    case Err::kTiWithoutSpirv:
      return "The taichi is not built with spirv";
    case Err::kCompiledKernelDataBroken:
      return "The CompiledKernelData is broken";
    case Err::kUnknown:
      return "Unknown error";
  }
  return "Unknown error";
}

std::unique_ptr<CompiledKernelData> CompiledKernelData::create(Arch arch,
                                                               Err &err) {
  err = Err::kUnknown;
  if (arch_uses_llvm(arch)) {
    if (llvm_creator) {
      err = Err::kNoError;
      return llvm_creator();
    } else {
      err = Err::kTiWithoutLLVM;
    }
  } else if (arch_uses_spirv(arch)) {
    if (spriv_creator) {
      err = Err::kNoError;
      return spriv_creator();
    } else {
      err = Err::kTiWithoutSpirv;
    }
  }
  return nullptr;
}

}  // namespace taichi::lang
