#include "taichi/transforms/make_external_shared_staged.h"  // implementation include

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"

namespace taichi::lang::irpass {
namespace {

bool constant_integer(Stmt *stmt, std::int64_t *value) {
  auto *constant = stmt ? stmt->cast<ConstStmt>() : nullptr;
  if (constant == nullptr || !is_integral(constant->val.dt)) {
    return false;
  }
  *value = constant->val.val_as_int64();
  return true;
}

bool checked_add(std::int64_t lhs, std::int64_t rhs, std::int64_t *result) {
  if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_subtract(std::int64_t lhs,
                      std::int64_t rhs,
                      std::int64_t *result) {
  if ((rhs > 0 && lhs < std::numeric_limits<std::int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<std::int64_t>::max() + rhs)) {
    return false;
  }
  *result = lhs - rhs;
  return true;
}

bool checked_align_up(std::int64_t value,
                      std::int64_t alignment,
                      std::int64_t *result) {
  if (value < 0 || alignment <= 0) {
    return false;
  }
  const auto remainder = value % alignment;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  return checked_add(value, alignment - remainder, result);
}

bool affine_loop_offset(Stmt *stmt,
                        OffloadedStmt *offload,
                        std::int64_t *offset) {
  if (auto *index = stmt ? stmt->cast<LoopIndexStmt>() : nullptr;
      index != nullptr && index->loop == offload && index->index == 0) {
    *offset = 0;
    return true;
  }
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  if (binary == nullptr) {
    return false;
  }
  std::int64_t base_offset = 0;
  std::int64_t constant = 0;
  if (binary->op_type == BinaryOpType::add) {
    if (affine_loop_offset(binary->lhs, offload, &base_offset) &&
        constant_integer(binary->rhs, &constant)) {
      return checked_add(base_offset, constant, offset);
    }
    if (constant_integer(binary->lhs, &constant) &&
        affine_loop_offset(binary->rhs, offload, &base_offset)) {
      return checked_add(constant, base_offset, offset);
    }
  }
  if (binary->op_type == BinaryOpType::sub &&
      affine_loop_offset(binary->lhs, offload, &base_offset) &&
      constant_integer(binary->rhs, &constant)) {
    return checked_subtract(base_offset, constant, offset);
  }
  return false;
}

ExternalPtrStmt *direct_external_pointer(Stmt *stmt) {
  return stmt ? stmt->cast<ExternalPtrStmt>() : nullptr;
}

ArgLoadStmt *external_base(ExternalPtrStmt *pointer) {
  return pointer ? pointer->base_ptr->cast<ArgLoadStmt>() : nullptr;
}

void require(bool condition, std::size_t task_index, const char *reason) {
  TI_ERROR_IF(!condition, "shared_staged_1d rejected physical task {}: {}",
              task_index, reason);
}

using ArgPath = std::vector<int>;

struct ExternalAccess {
  ExternalPtrStmt *pointer{nullptr};
  ArgLoadStmt *base{nullptr};
  DataType data_type;
  std::vector<int> element_shape;
  int lane_index{0};
  int lane_count{1};
  std::int64_t offset{0};
  bool read{false};
  bool write{false};
};

int compound_lane_count(const std::vector<int> &shape,
                        std::size_t task_index) {
  std::int64_t count = 1;
  require(shape.size() <= 2, task_index,
          "compound elements must be rank-one vectors or rank-two matrices");
  for (int extent : shape) {
    require(extent > 0 && count <= 16 / extent, task_index,
            "compound elements exceed the sixteen-lane prototype bound");
    count *= extent;
  }
  return static_cast<int>(count);
}

std::vector<int> compound_lane_coordinates(
    int lane_index,
    const std::vector<int> &element_shape) {
  std::vector<int> result(element_shape.size());
  for (std::size_t reverse_axis = 0; reverse_axis < element_shape.size();
       ++reverse_axis) {
    const std::size_t axis = element_shape.size() - 1 - reverse_axis;
    result[axis] = lane_index % element_shape[axis];
    lane_index /= element_shape[axis];
  }
  TI_ASSERT(lane_index == 0);
  return result;
}

ExternalAccess describe_external_access(ExternalPtrStmt *pointer,
                                        OffloadedStmt *offload,
                                        bool read,
                                        bool write,
                                        const std::map<ArgPath, std::vector<int>>
                                            &semantic_element_shapes,
                                        std::size_t task_index) {
  ExternalAccess result;
  result.pointer = pointer;
  result.base = external_base(pointer);
  result.read = read;
  result.write = write;
  require(result.base != nullptr && result.base->is_ptr, task_index,
          "access is not backed by a direct ndarray argument");
  require(result.base->arg_id.size() == 1 && result.base->arg_id.front() >= 0,
          task_index, "only top-level ndarray arguments are supported");
  require(pointer->ndim == 1 &&
              pointer->indices.size() == 1 + pointer->element_shape.size() &&
              !pointer->is_grad &&
              pointer->boundary == BoundaryMode::kUnsafe,
          task_index,
          "only unsafe-boundary one-dimensional scalar, vector, or matrix "
          "ndarray accesses are supported");
  require(affine_loop_offset(pointer->indices.front(), offload, &result.offset),
          task_index, "external index is not loop_index plus a constant");
  const auto semantic_shape = semantic_element_shapes.find(result.base->arg_id);
  result.element_shape =
      semantic_shape == semantic_element_shapes.end()
          ? pointer->element_shape
          : semantic_shape->second;
  result.lane_count = compound_lane_count(result.element_shape, task_index);
  require(compound_lane_count(pointer->element_shape, task_index) ==
              result.lane_count,
          task_index,
          "scalarized compound lane count disagrees with its symbolic shape");
  for (std::size_t axis = 0; axis < pointer->element_shape.size(); ++axis) {
    std::int64_t coordinate = 0;
    require(constant_integer(pointer->indices[1 + axis], &coordinate) &&
                coordinate >= 0 && coordinate < pointer->element_shape[axis],
            task_index, "compound element lane is not a bounded constant");
    result.lane_index =
        result.lane_index * pointer->element_shape[axis] + coordinate;
  }
  result.data_type = pointer->ret_type.ptr_removed();
  require(result.data_type->is<PrimitiveType>(), task_index,
          "compound accesses must be scalarized to primitive lanes");
  const auto scalar_bytes = data_type_size(result.data_type);
  require(scalar_bytes == 2 || scalar_bytes == 4 || scalar_bytes == 8,
          task_index,
          "only two-, four-, or eight-byte primitive lanes are supported");
  return result;
}

Stmt *push_add_constant(Block *block, Stmt *value, std::int64_t constant) {
  if (constant == 0) {
    return value;
  }
  return block->push_back<BinaryOpStmt>(
      BinaryOpType::add, value,
      block->push_back<ConstStmt>(TypedConstant(static_cast<int32>(constant))));
}

void materialize_shared_task(OffloadedStmt *offload,
                             const Kernel::OffloadTaskOptimizationSpec &spec,
                             std::size_t task_index) {
  require(offload->task_type == OffloadedStmt::TaskType::range_for, task_index,
          "strategy applies only to range_for tasks");
  require(offload->const_begin && offload->const_end &&
              offload->end_value > offload->begin_value,
          task_index,
          "iteration range must be a non-empty compile-time constant");
  require(!offload->reversed && !offload->one_to_one &&
              offload->bls_prologue == nullptr &&
              offload->bls_epilogue == nullptr,
          task_index, "task already has an incompatible range or BLS mapping");
  require(spec.workgroup_size > 0 && offload->block_dim == spec.workgroup_size,
          task_index, "exact requested workgroup size was not materialized");

  bool has_compound_access = false;
  std::map<ArgPath, std::vector<int>> semantic_element_shapes;
  analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
    if (auto *pointer = stmt->cast<ExternalPtrStmt>()) {
      has_compound_access |= !pointer->element_shape.empty();
      if (!pointer->element_shape.empty()) {
        auto *base = external_base(pointer);
        require(base != nullptr, task_index,
                "compound access has no ndarray argument owner");
        const auto [entry, inserted] = semantic_element_shapes.emplace(
            base->arg_id, pointer->element_shape);
        require(inserted || entry->second == pointer->element_shape, task_index,
                "compound argument has inconsistent symbolic element shapes");
      }
    }
    return false;
  });
  if (has_compound_access) {
    // A compound ndarray is physically staged as a packed AoS record of
    // primitive lanes. Scalarizing before access analysis avoids relying on
    // target-specific LLVM vector alignment for 3-lane or matrix records.
    require(irpass::scalarize(offload, false), task_index,
            "compound element scalarization did not materialize lane accesses");
    irpass::die(offload);
  }

  std::vector<ExternalPtrStmt *> external_pointers;
  std::vector<GlobalLoadStmt *> loads;
  std::vector<GlobalStoreStmt *> stores;
  bool nested_control = false;
  bool unsupported_side_effect = false;
  analysis::gather_statements(offload->body.get(), [&](Stmt *stmt) {
    if (stmt->is_container_statement()) {
      nested_control = true;
    }
    if (auto *pointer = stmt->cast<ExternalPtrStmt>()) {
      external_pointers.push_back(pointer);
    } else if (auto *load = stmt->cast<GlobalLoadStmt>()) {
      loads.push_back(load);
    } else if (auto *store = stmt->cast<GlobalStoreStmt>()) {
      stores.push_back(store);
    } else if (stmt->has_global_side_effect()) {
      unsupported_side_effect = true;
    }
    return false;
  });
  require(!nested_control, task_index, "nested control flow is unsupported");
  require(!unsupported_side_effect, task_index,
          "task contains an unsupported global side effect");
  require(!loads.empty() && !stores.empty(), task_index,
          "recipe requires external reads and pointwise external writes");

  std::vector<ExternalAccess> accesses;
  std::set<ExternalPtrStmt *> consumed_pointers;
  for (auto *load : loads) {
    auto *pointer = direct_external_pointer(load->src);
    require(pointer != nullptr, task_index,
            "all global loads must use direct external pointers");
    accesses.push_back(
        describe_external_access(pointer, offload, true, false,
                                 semantic_element_shapes, task_index));
    consumed_pointers.insert(pointer);
  }
  for (auto *store : stores) {
    auto *pointer = direct_external_pointer(store->dest);
    require(pointer != nullptr, task_index,
            "all global stores must use direct external pointers");
    accesses.push_back(
        describe_external_access(pointer, offload, false, true,
                                 semantic_element_shapes, task_index));
    consumed_pointers.insert(pointer);
  }
  require(consumed_pointers.size() == external_pointers.size(), task_index,
          "external pointers must be consumed directly by one load or store");

  struct ArgumentSummary {
    bool reads{false};
    bool writes{false};
    DataType data_type;
    std::vector<int> element_shape;
    int lane_count{1};
    ArgLoadStmt *base{nullptr};
    std::set<std::int64_t> read_offsets;
    std::set<std::int64_t> write_offsets;
    std::map<std::int64_t, std::set<int>> read_lanes;
    std::map<std::int64_t, std::set<int>> write_lanes;
  };
  std::map<ArgPath, ArgumentSummary> arguments;
  for (const auto &access : accesses) {
    auto &summary = arguments[access.base->arg_id];
    if (summary.base == nullptr) {
      summary.base = access.base;
      summary.data_type = access.data_type;
      summary.element_shape = access.element_shape;
      summary.lane_count = access.lane_count;
    }
    require(summary.data_type == access.data_type &&
                summary.element_shape == access.element_shape &&
                summary.lane_count == access.lane_count,
            task_index,
            "one ndarray argument is accessed with inconsistent element ABI");
    summary.reads = summary.reads || access.read;
    summary.writes = summary.writes || access.write;
    if (access.read) {
      summary.read_offsets.insert(access.offset);
      summary.read_lanes[access.offset].insert(access.lane_index);
    }
    if (access.write) {
      summary.write_offsets.insert(access.offset);
      summary.write_lanes[access.offset].insert(access.lane_index);
    }
  }

  struct InputTile {
    const ArgumentSummary *summary{nullptr};
    ArgPath path;
    std::int64_t halo_low{0};
    std::int64_t halo_high{0};
    std::int64_t element_bytes{0};
    std::int64_t scalar_bytes{0};
    std::int64_t alignment{0};
    std::int64_t tile_elements{0};
    std::int64_t byte_offset{0};
    std::int64_t tile_bytes{0};
  };
  std::vector<InputTile> input_tiles;
  const std::set<int> requested_source_indices(
      spec.memory_source_arg_indices.begin(),
      spec.memory_source_arg_indices.end());
  std::set<int> materialized_source_indices;
  std::int64_t aggregate_tile_bytes = 0;
  for (const auto &[path, summary] : arguments) {
    std::set<int> complete_lanes;
    for (int lane = 0; lane < summary.lane_count; ++lane) {
      complete_lanes.insert(lane);
    }
    if (summary.reads) {
      require(!summary.writes, task_index,
              "staged inputs must be read-only external ndarrays");
      const bool selected = requested_source_indices.empty() ||
                            requested_source_indices.count(path.front()) != 0;
      if (!selected) {
        continue;
      }
      require(summary.read_offsets.size() >= 2, task_index,
              "each read-only input must have at least two distinct affine "
              "offsets");
      require(std::all_of(summary.read_lanes.begin(),
                          summary.read_lanes.end(),
                          [&](const auto &item) {
                            return item.second == complete_lanes;
                          }),
              task_index,
              "compound staged inputs must read every lane at every offset");
      const std::int64_t halo_low = *summary.read_offsets.begin();
      const std::int64_t halo_high = *summary.read_offsets.rbegin();
      require(halo_low < halo_high && halo_low >= -4096 && halo_high <= 4096,
              task_index,
              "stencil halo is empty or outside the prototype bound");
      const std::int64_t tile_elements =
          static_cast<std::int64_t>(offload->block_dim) + halo_high - halo_low;
      const std::int64_t scalar_bytes = data_type_size(summary.data_type);
      const std::int64_t element_bytes =
          scalar_bytes * static_cast<std::int64_t>(summary.lane_count);
      const std::int64_t alignment = data_type_alignment(summary.data_type);
      std::int64_t byte_offset = 0;
      require(checked_align_up(aggregate_tile_bytes, alignment, &byte_offset),
              task_index, "shared tile alignment overflowed");
      require(tile_elements > 0 &&
                  tile_elements <=
                      std::numeric_limits<std::int64_t>::max() / element_bytes,
              task_index, "shared tile size overflowed");
      const std::int64_t tile_bytes = tile_elements * element_bytes;
      require(checked_add(byte_offset, tile_bytes, &aggregate_tile_bytes) &&
                  aggregate_tile_bytes <= 32 * 1024,
              task_index,
              "aggregate shared tiles exceed the 32 KiB prototype budget");
      input_tiles.push_back(InputTile{&summary, path, halo_low, halo_high,
                                      element_bytes, scalar_bytes, alignment,
                                      tile_elements, byte_offset, tile_bytes});
      materialized_source_indices.insert(path.front());
    } else {
      require(
          summary.writes && summary.write_offsets == std::set<std::int64_t>{0},
          task_index, "outputs must be write-only and pointwise");
      require(summary.write_lanes.size() == 1 &&
                  summary.write_lanes.begin()->second == complete_lanes,
              task_index, "compound outputs must write every lane pointwise");
    }
  }
  require(!input_tiles.empty() && input_tiles.size() <= 2, task_index,
          "recipe requires one or two read-only affine inputs");
  require(requested_source_indices.empty() ||
              materialized_source_indices == requested_source_indices,
          task_index,
          "requested staged source is not an eligible read-only stencil input");

  offload->bls_prologue = std::make_unique<Block>();
  offload->bls_prologue->set_parent_stmt(offload);
  auto *prologue = offload->bls_prologue.get();
  auto *thread_index = prologue->push_back<LoopLinearIndexStmt>(offload);
  for (const auto &tile : input_tiles) {
    const auto source_access = std::find_if(
        accesses.begin(), accesses.end(), [&](const ExternalAccess &access) {
          return access.read && access.base->arg_id == tile.path;
        });
    TI_ASSERT(source_access != accesses.end());
    const auto *source_pointer = source_access->pointer;
    for (std::int64_t loop_offset = 0; loop_offset < tile.tile_elements;
         loop_offset += offload->block_dim) {
      auto *tile_index = push_add_constant(prologue, thread_index, loop_offset);
      auto *global_index = prologue->push_back<BinaryOpStmt>(
          BinaryOpType::add,
          prologue->push_back<BlockCornerIndexStmt>(offload, 0), tile_index);
      global_index = push_add_constant(prologue, global_index, tile.halo_low);

      Stmt *condition = prologue->push_back<BinaryOpStmt>(
          BinaryOpType::cmp_ge, global_index,
          prologue->push_back<ConstStmt>(TypedConstant(static_cast<int32>(
              static_cast<std::int64_t>(offload->begin_value) +
              tile.halo_low))));
      auto *upper = prologue->push_back<BinaryOpStmt>(
          BinaryOpType::cmp_le, global_index,
          prologue->push_back<ConstStmt>(TypedConstant(static_cast<int32>(
              static_cast<std::int64_t>(offload->end_value) - 1 +
              tile.halo_high))));
      condition = prologue->push_back<BinaryOpStmt>(BinaryOpType::bit_and,
                                                    condition, upper);
      if (loop_offset + offload->block_dim > tile.tile_elements) {
        auto *within_tile = prologue->push_back<BinaryOpStmt>(
            BinaryOpType::cmp_lt, tile_index,
            prologue->push_back<ConstStmt>(
                TypedConstant(static_cast<int32>(tile.tile_elements))));
        condition = prologue->push_back<BinaryOpStmt>(BinaryOpType::bit_and,
                                                      condition, within_tile);
      }
      auto *guard = prologue->push_back<IfStmt>(condition)->as<IfStmt>();
      guard->set_true_statements(std::make_unique<Block>());
      auto *body = guard->true_statements.get();
      auto *base = body->push_back<ArgLoadStmt>(
          tile.path, tile.summary->base->ret_type, tile.summary->base->is_ptr,
          tile.summary->base->create_load, tile.summary->base->arg_depth);
      for (int lane = 0; lane < tile.summary->lane_count; ++lane) {
        std::vector<Stmt *> indices{global_index};
        for (int coordinate : compound_lane_coordinates(
                 lane, source_pointer->element_shape)) {
          indices.push_back(body->push_back<ConstStmt>(
              TypedConstant(static_cast<int32>(coordinate))));
        }
        auto *pointer = body->push_back<ExternalPtrStmt>(
            base, indices, 1, source_pointer->element_shape, false,
            BoundaryMode::kUnsafe, source_pointer->byte_offset,
            source_pointer->byte_stride, source_pointer->runtime_affine);
        auto *value = body->push_back<GlobalLoadStmt>(pointer);
        Stmt *byte_offset = body->push_back<BinaryOpStmt>(
            BinaryOpType::mul, tile_index,
            body->push_back<ConstStmt>(
                TypedConstant(static_cast<int32>(tile.element_bytes))));
        byte_offset = push_add_constant(
            body, byte_offset,
            tile.byte_offset + lane * tile.scalar_bytes);
        auto input_data_type = tile.summary->data_type;
        auto *local_pointer = body->push_back<BlockLocalPtrStmt>(
            byte_offset,
            TypeFactory::get_instance().get_pointer_type(input_data_type));
        body->push_back<GlobalStoreStmt>(local_pointer, value);
      }
    }
  }

  for (const auto &access : accesses) {
    if (!access.read) {
      continue;
    }
    const auto tile = std::find_if(
        input_tiles.begin(), input_tiles.end(), [&](const InputTile &candidate) {
          return access.base->arg_id == candidate.path;
        });
    if (tile == input_tiles.end()) {
      // This read belongs to an eligible stencil source that the complete
      // Graph recipe intentionally leaves on the direct global-memory path.
      continue;
    }
    VecStatement replacement;
    Stmt *local_index = replacement.push_back<LoopLinearIndexStmt>(offload);
    local_index = replacement.push_back<BinaryOpStmt>(
        BinaryOpType::add, local_index,
        replacement.push_back<ConstStmt>(
            TypedConstant(static_cast<int32>(access.offset - tile->halo_low))));
    Stmt *byte_offset = replacement.push_back<BinaryOpStmt>(
        BinaryOpType::mul, local_index,
        replacement.push_back<ConstStmt>(
            TypedConstant(static_cast<int32>(tile->element_bytes))));
    const auto lane_byte_offset =
        tile->byte_offset + access.lane_index * tile->scalar_bytes;
    if (lane_byte_offset != 0) {
      byte_offset = replacement.push_back<BinaryOpStmt>(
          BinaryOpType::add, byte_offset,
          replacement.push_back<ConstStmt>(
              TypedConstant(static_cast<int32>(lane_byte_offset))));
    }
    auto input_data_type = tile->summary->data_type;
    replacement.push_back<BlockLocalPtrStmt>(
        byte_offset,
        TypeFactory::get_instance().get_pointer_type(input_data_type));
    access.pointer->replace_with(std::move(replacement));
  }

  const std::int64_t range_size =
      static_cast<std::int64_t>(offload->end_value) - offload->begin_value;
  offload->grid_dim = static_cast<int>((range_size + offload->block_dim - 1) /
                                       offload->block_dim);
  offload->bls_size = static_cast<std::size_t>(aggregate_tile_bytes);
  offload->external_shared_staged = true;
  offload->external_shared_arg_index = input_tiles.front().path.front();
  offload->external_shared_halo_low =
      static_cast<int>(input_tiles.front().halo_low);
  offload->external_shared_halo_high =
      static_cast<int>(input_tiles.front().halo_high);
  for (const auto &tile : input_tiles) {
    offload->external_shared_arg_indices.push_back(tile.path.front());
    offload->external_shared_halo_lows.push_back(
        static_cast<int>(tile.halo_low));
    offload->external_shared_halo_highs.push_back(
        static_cast<int>(tile.halo_high));
    offload->external_shared_byte_offsets.push_back(
        static_cast<int>(tile.byte_offset));
    offload->external_shared_element_bytes.push_back(
        static_cast<int>(tile.element_bytes));
    offload->external_shared_scalar_bytes.push_back(
        static_cast<int>(tile.scalar_bytes));
    offload->external_shared_element_shapes.push_back(
        tile.summary->element_shape);
  }
}

}  // namespace

void make_external_shared_staged(IRNode *ir,
                                 const CompileConfig &config,
                                 const Kernel *kernel) {
  const auto &plan = kernel->get_offload_execution_plan();
  if (!plan.has_value()) {
    return;
  }
  TI_ERROR_IF(config.arch != Arch::cuda,
              "shared_staged_1d requires the CUDA backend");
  TI_ERROR_IF(!ir->is<Block>(),
              "shared_staged_1d expected an offloaded kernel block");
  std::size_t task_index = 0;
  for (const auto &statement : ir->as<Block>()->statements) {
    auto *offload = statement->cast<OffloadedStmt>();
    if (offload == nullptr) {
      continue;
    }
    const auto &spec =
        kernel->offload_task_optimization_spec(task_index, offload->task_type);
    if (spec.memory_strategy == "shared_staged_1d") {
      materialize_shared_task(offload, spec, task_index);
    }
    ++task_index;
  }
}

}  // namespace taichi::lang::irpass
