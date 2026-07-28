#include "taichi/python/export_storage_view.h"

#include <string>
#include <vector>

#include "taichi/ir/snode.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"

namespace taichi {
namespace {

lang::storage::StorageAccess parse_storage_access(const std::string &access) {
  using lang::storage::StorageAccess;
  if (access == "readonly" || access == "read") {
    return StorageAccess::kReadOnly;
  }
  if (access == "writeonly" || access == "write") {
    return StorageAccess::kWriteOnly;
  }
  if (access == "readwrite") {
    return StorageAccess::kReadWrite;
  }
  throw py::value_error(
      "storage access must be 'readonly', 'writeonly', or 'readwrite'");
}

lang::storage::RuntimeStorageConsumer parse_runtime_storage_consumer(
    const std::string &consumer) {
  using lang::storage::RuntimeStorageConsumer;
  if (consumer == "ordinary_kernel") {
    return RuntimeStorageConsumer::kOrdinaryKernel;
  }
  if (consumer == "graph_replay") {
    return RuntimeStorageConsumer::kGraphReplay;
  }
  if (consumer == "graph_capture") {
    return RuntimeStorageConsumer::kGraphCapture;
  }
  if (consumer == "native_consumer") {
    return RuntimeStorageConsumer::kNativeConsumer;
  }
  if (consumer == "external_interop") {
    return RuntimeStorageConsumer::kExternalInterop;
  }
  throw py::value_error("unsupported runtime storage consumer");
}

lang::storage::RuntimeStorageMode parse_runtime_storage_mode(
    const std::string &mode) {
  using lang::storage::RuntimeStorageMode;
  if (mode == "ordinary") {
    return RuntimeStorageMode::kOrdinary;
  }
  if (mode == "replay") {
    return RuntimeStorageMode::kReplay;
  }
  if (mode == "capture") {
    return RuntimeStorageMode::kCapture;
  }
  throw py::value_error(
      "runtime storage mode must be ordinary, replay, or capture");
}

py::dict dense_storage_properties(
    const lang::storage::DenseStorageDescriptor &descriptor) {
  using lang::storage::to_string;
  const auto &properties = descriptor.properties();
  py::dict result;
  result["empty"] = properties.empty;
  result["aligned"] = properties.aligned;
  result["compact_contiguous"] = properties.compact_contiguous;
  result["element_contiguous"] = properties.element_contiguous;
  result["canonical_aos"] = properties.canonical_aos;
  result["canonical_soa"] = properties.canonical_soa;
  result["ndarray_abi_compatible"] = properties.ndarray_abi_compatible;
  result["single_record_stride_compatible"] =
      properties.single_record_stride_compatible;
  result["has_negative_stride"] = properties.has_negative_stride;
  result["scalar_size"] = properties.scalar_size;
  result["scalar_alignment"] = properties.scalar_alignment;
  result["scalar_count"] = properties.scalar_count;
  result["item_count"] = properties.item_count;
  result["reachable_begin"] = properties.reachable_begin;
  result["reachable_end"] = properties.reachable_end;
  result["record_stride"] = properties.record_stride;
  result["array_layout"] = to_string(properties.array_layout);
  result["uniqueness"] = to_string(properties.uniqueness);
  return result;
}

py::dict runtime_storage_qualification_dict(
    const lang::storage::RuntimeStorageQualification &qualification) {
  py::dict result;
  result["describable"] = qualification.capabilities.describable;
  result["bindable"] = qualification.capabilities.bindable;
  result["replayable"] = qualification.capabilities.replayable;
  result["capturable"] = qualification.capabilities.capturable;
  result["zero_copy_qualified"] =
      qualification.capabilities.zero_copy_qualified;
  result["dense_supported"] = qualification.dense.supported;
  result["execution_mode"] =
      lang::storage::to_string(qualification.dense.execution_mode);
  result["reason"] = lang::storage::to_string(qualification.reason);
  result["stable_signature"] = qualification.stable_signature;
  return result;
}

py::dict storage_qualification_dict(
    const lang::storage::StorageQualification &qualification) {
  py::dict result;
  result["supported"] = qualification.supported;
  result["execution_mode"] =
      lang::storage::to_string(qualification.execution_mode);
  result["reason"] = lang::storage::to_string(qualification.reason);
  result["requires_materialization"] = qualification.requires_materialization;
  result["estimated_copy_bytes"] = qualification.estimated_copy_bytes;
  return result;
}

}  // namespace

void export_storage_view(py::module &m) {
  using namespace lang;
  using namespace lang::storage;

  py::class_<DenseStorageDescriptor>(m, "_DenseStorageDescriptor")
      .def_property_readonly("owner_kind",
                             [](const DenseStorageDescriptor &descriptor) {
                               return to_string(descriptor.owner().kind);
                             })
      .def_property_readonly("source_kind",
                             [](const DenseStorageDescriptor &descriptor) {
                               return to_string(descriptor.source_kind());
                             })
      .def_property_readonly("scalar_type",
                             &DenseStorageDescriptor::scalar_type)
      .def_property_readonly("access",
                             [](const DenseStorageDescriptor &descriptor) {
                               return to_string(descriptor.access());
                             })
      .def_property_readonly("index_shape",
                             &DenseStorageDescriptor::index_shape)
      .def_property_readonly("index_strides_bytes",
                             &DenseStorageDescriptor::index_strides_bytes)
      .def_property_readonly("element_shape",
                             &DenseStorageDescriptor::element_shape)
      .def_property_readonly("element_strides_bytes",
                             &DenseStorageDescriptor::element_strides_bytes)
      .def_property_readonly("byte_offset",
                             &DenseStorageDescriptor::byte_offset)
      .def_property_readonly("fingerprint",
                             &DenseStorageDescriptor::fingerprint)
      .def_property_readonly("program_domain",
                             [](const DenseStorageDescriptor &descriptor) {
                               return descriptor.owner().program_domain;
                             })
      .def_property_readonly(
          "resource_identity",
          [](const DenseStorageDescriptor &descriptor) -> py::object {
            const auto &owner = descriptor.owner();
            if (owner.kind != StorageOwnerKind::kProgramNdarray) {
              return py::none();
            }
            const auto &handle = owner.ndarray_handle;
            return py::make_tuple(handle.domain, handle.kind, handle.index,
                                  handle.generation);
          })
      .def_property_readonly(
          "tree_identity",
          [](const DenseStorageDescriptor &descriptor) -> py::object {
            const auto &owner = descriptor.owner();
            if (owner.kind != StorageOwnerKind::kSNodePayload) {
              return py::none();
            }
            return py::make_tuple(owner.tree.tree_id, owner.tree.generation,
                                  owner.tree.layout_fingerprint);
          })
      .def_property_readonly("anchor_snode_id",
                             [](const DenseStorageDescriptor &descriptor) {
                               return descriptor.owner().anchor_snode_id;
                             })
      .def_property_readonly("component_snode_ids",
                             [](const DenseStorageDescriptor &descriptor) {
                               return descriptor.owner().component_snode_ids;
                             })
      .def_property_readonly("properties", &dense_storage_properties);

  py::class_<RuntimeStorageArgument>(m, "_RuntimeStorageArgument")
      .def_property_readonly("descriptor", &RuntimeStorageArgument::descriptor,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("stable_signature",
                             &RuntimeStorageArgument::stable_signature)
      .def_property_readonly(
          "synchronization_domain_identity",
          &RuntimeStorageArgument::synchronization_domain_identity)
      .def_property_readonly(
          "consumer",
          [](const RuntimeStorageArgument &argument) {
            return to_string(argument.requirement().consumer);
          })
      .def_property_readonly("mode",
                             [](const RuntimeStorageArgument &argument) {
                               return to_string(argument.requirement().mode);
                             })
      .def_property_readonly(
          "qualification", [](const RuntimeStorageArgument &argument) {
            return runtime_storage_qualification_dict(argument.qualification());
          });

  py::class_<DenseStorageBuildResult>(m, "_DenseStorageBuildResult")
      .def_property_readonly("ok",
                             [](const DenseStorageBuildResult &result) {
                               return static_cast<bool>(result);
                             })
      .def_property_readonly("reason",
                             [](const DenseStorageBuildResult &result) {
                               return to_string(result.reason);
                             })
      .def_property_readonly(
          "descriptor",
          [](const DenseStorageBuildResult &result)
              -> const DenseStorageDescriptor * {
            return result.descriptor ? &*result.descriptor : nullptr;
          },
          py::return_value_policy::reference_internal);

  m.def(
      "_make_runtime_storage_argument",
      [](Program &program, const DenseStorageDescriptor &descriptor,
         const std::string &consumer, const std::string &mode,
         std::uint64_t synchronization_domain_identity,
         bool require_external_sync) {
        RuntimeStorageRequirement requirement;
        requirement.dense.max_element_rank = 2;
        requirement.dense.require_ndarray_abi = false;
        requirement.dense.accept_general_affine = true;
        requirement.dense.require_unique_mapping = true;
        requirement.dense.require_writable = true;
        requirement.dense.accept_external_owner = true;
        requirement.dense.allow_materialization = false;
        requirement.consumer = parse_runtime_storage_consumer(consumer);
        requirement.mode = parse_runtime_storage_mode(mode);
        requirement.backend = program.compile_config().arch;
        requirement.require_external_sync = require_external_sync;
        return RuntimeStorageArgument(descriptor, requirement,
                                      synchronization_domain_identity);
      },
      py::arg("program"), py::arg("descriptor"),
      py::arg("consumer") = "ordinary_kernel", py::arg("mode") = "ordinary",
      py::arg("synchronization_domain_identity") = 0,
      py::arg("require_external_sync") = false);

  m.def(
      "_describe_ndarray_storage",
      [](const Ndarray &array, const std::string &access) {
        return describe_ndarray_storage(array, parse_storage_access(access));
      },
      py::arg("array"), py::arg("access") = "readwrite");

  m.def("_flatten_dense_storage_to_scalar_vector",
        &flatten_dense_storage_to_scalar_vector,
        py::arg("descriptor"));

  m.def("_slice_dense_storage", &slice_dense_storage,
        py::arg("descriptor"), py::arg("starts"), py::arg("lengths"),
        py::arg("steps"));

  m.def(
      "_describe_struct_member_storage",
      [](const Ndarray &base, DataType scalar_type,
         const std::vector<std::int64_t> &index_shape,
         const std::vector<std::int64_t> &element_shape,
         std::int64_t byte_offset, std::int64_t record_stride,
         bool tensor_member, const std::string &access) {
        return describe_struct_member_storage(
            base, scalar_type, index_shape, element_shape, byte_offset,
            record_stride,
            tensor_member ? StorageSourceKind::kStructTensorMember
                          : StorageSourceKind::kStructScalarMember,
            parse_storage_access(access));
      },
      py::arg("base"), py::arg("scalar_type"), py::arg("index_shape"),
      py::arg("element_shape"), py::arg("byte_offset"),
      py::arg("record_stride"), py::arg("tensor_member"),
      py::arg("access") = "readwrite");

  m.def(
      "_describe_dense_field_storage",
      [](Program &program, SNode *anchor,
         const std::vector<SNode *> &components, DataType scalar_type,
         const std::vector<std::int64_t> &index_shape,
         const std::vector<std::int64_t> &element_shape,
         const std::string &access) {
        return describe_dense_field_storage(
            program, anchor, components, scalar_type, index_shape,
            element_shape, parse_storage_access(access));
      },
      py::arg("program"), py::arg("anchor"), py::arg("components"),
      py::arg("scalar_type"), py::arg("index_shape"), py::arg("element_shape"),
      py::arg("access") = "readwrite");

  m.def(
      "_validate_storage_owner",
      [](Program &program, const DenseStorageDescriptor &descriptor) {
        return to_string(validate_storage_owner(program, descriptor));
      },
      py::arg("program"), py::arg("descriptor"));

  m.def(
      "_analyze_storage_alias",
      [](const DenseStorageDescriptor &lhs, const DenseStorageDescriptor &rhs) {
        return to_string(analyze_logical_storage_alias(lhs, rhs));
      },
      py::arg("lhs"), py::arg("rhs"));

  m.def(
      "_qualify_dense_storage",
      [](const DenseStorageDescriptor &descriptor,
         const py::object &scalar_type, std::size_t min_index_rank,
         std::size_t max_index_rank, std::size_t max_element_rank,
         bool require_ndarray_abi, bool accept_compact_subrange,
         bool accept_single_record_stride, bool accept_general_affine,
         bool require_unique_mapping, bool require_writable,
         bool accept_external_owner, bool allow_materialization) {
        DenseStorageRequirement requirement;
        if (!scalar_type.is_none()) {
          requirement.require_scalar_type = true;
          requirement.scalar_type = scalar_type.cast<DataType>();
        }
        requirement.min_index_rank = min_index_rank;
        requirement.max_index_rank = max_index_rank;
        requirement.max_element_rank = max_element_rank;
        requirement.require_ndarray_abi = require_ndarray_abi;
        requirement.accept_compact_subrange = accept_compact_subrange;
        requirement.accept_single_record_stride = accept_single_record_stride;
        requirement.accept_general_affine = accept_general_affine;
        requirement.require_unique_mapping = require_unique_mapping;
        requirement.require_writable = require_writable;
        requirement.accept_external_owner = accept_external_owner;
        requirement.allow_materialization = allow_materialization;
        return storage_qualification_dict(
            qualify_dense_storage(descriptor, requirement));
      },
      py::arg("descriptor"), py::arg("scalar_type") = py::none(),
      py::arg("min_index_rank") = 0,
      py::arg("max_index_rank") = kMaxDenseStorageRank,
      py::arg("max_element_rank") = kMaxDenseStorageRank,
      py::arg("require_ndarray_abi") = false,
      py::arg("accept_compact_subrange") = true,
      py::arg("accept_single_record_stride") = false,
      py::arg("accept_general_affine") = false,
      py::arg("require_unique_mapping") = false,
      py::arg("require_writable") = false,
      py::arg("accept_external_owner") = false,
      py::arg("allow_materialization") = false);
}

}  // namespace taichi
