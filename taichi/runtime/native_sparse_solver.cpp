#include "taichi/runtime/native_sparse_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/ir/type_utils.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_csr_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/program/sparse_solver.h"
#include "taichi/rhi/cpu/cpu_device.h"
#endif

namespace taichi::lang {
namespace {

void validate_policy(const SparseDirectPlanDescription &d) {
  if (!d.rows || d.rows > std::uint64_t(std::numeric_limits<int>::max()) ||
      d.nonzeros > std::uint64_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument(
        "Sparse direct dimensions and stored counts must fit the i32 core");
  if ((d.dtype != PrimitiveType::f32 && d.dtype != PrimitiveType::f64) ||
      (d.method != "LLT" && d.method != "LDLT" && d.method != "LU") ||
      (d.ordering != "AMD" && d.ordering != "COLAMD") ||
      (d.format != "csr" && d.format != "csc") ||
      d.implementation != "eigen")
    throw std::invalid_argument("Unsupported sparse direct numerical policy");
}

void validate_vector(Device &device,
                     const NativeArrayBinding &array,
                     DataType dtype,
                     std::uint64_t size) {
  const auto width = std::uint64_t(data_type_size(dtype));
  if (array.pointer.device != &device || array.scalar_type != dtype ||
      size > std::numeric_limits<std::size_t>::max() / width ||
      array.bytes != size * width)
    throw std::invalid_argument(
        "Sparse direct vectors require matching dtype, count and device");
  validate_native_array_layout(array, {{size}, {}, width, width, true});
}

void validate_plan(Device &device, const SparseDirectPlanDescription &d) {
  validate_policy(d);
  validate_vector(device, d.arrays[0], PrimitiveType::i32, d.rows + 1);
  validate_vector(device, d.arrays[1], PrimitiveType::i32, d.nonzeros);
  validate_vector(device, d.arrays[2], d.dtype, d.nonzeros);
}

void validate_solve(Device &device, const SparseDirectSolveOperation &op) {
  if (!op.plan || &op.plan->device() != &device || op.rhs_outputs.empty())
    throw std::invalid_argument(
        "Sparse direct solve requires a local plan and a nonempty RHS group");
  const auto &d = op.plan->description();
  if (d.refactor != bool(op.values))
    throw std::invalid_argument(
        "Sparse direct values binding must match the plan numerical phase");
  std::vector<NativeStorageRange> reads;
  std::vector<NativeStorageWrite> writes;
  auto range = [&](const NativeArrayBinding &array, std::uint64_t count) {
    validate_vector(device, array, d.dtype, count);
    return resolve_native_storage_range(op.plan->arch(), array.pointer,
                                         array.bytes);
  };
  if (op.values)
    reads.push_back(range(*op.values, d.nonzeros));
  for (const auto &[rhs, output] : op.rhs_outputs) {
    reads.push_back(range(rhs, d.rows));
    writes.push_back({range(output, d.rows), std::nullopt});
  }
  validate_native_disjoint_writes(reads, writes);
}

SparseDirectPlanDescription decode_plan(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"rows", "nonzeros", "dtype", "implementation", "method",
             "ordering", "format", "phase"});
  SparseDirectPlanDescription d;
  d.rows = integer(p["rows"]);
  d.nonzeros = integer(p["nonzeros"]);
  const auto dtype = name(p["dtype"]);
  if (dtype != "f32" && dtype != "f64")
    throw std::invalid_argument("Sparse direct dtype must be f32 or f64");
  d.dtype = dtype == "f32" ? PrimitiveType::f32 : PrimitiveType::f64;
  d.implementation = name(p["implementation"]);
  d.method = name(p["method"]);
  d.ordering = name(p["ordering"]);
  d.format = name(p["format"]);
  const auto phase = name(p["phase"]);
  if (phase != "solve" && phase != "refactor_solve")
    throw std::invalid_argument("Unsupported sparse direct numerical phase");
  d.refactor = phase == "refactor_solve";
  validate_policy(d);
  return d;
}

#ifdef TI_WITH_LLVM
void *cpu_pointer(Device &device, const NativeArrayBinding &array) {
  auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device);
  void *result = nullptr;
  if (!cpu ||
      cpu->map_range_for_cpu_native(array.pointer, array.bytes, &result) !=
          RhiResult::success)
    throw std::invalid_argument("Sparse direct CPU storage range is invalid");
  return result;
}

template <typename Scalar>
class OwnedEigenMatrix final
    : public EigenSparseMatrix<Eigen::SparseMatrix<Scalar>> {
 public:
  using Matrix = Eigen::SparseMatrix<Scalar>;
  using Base = EigenSparseMatrix<Matrix>;
  explicit OwnedEigenMatrix(const Matrix &matrix) : Base(matrix) {
  }
  void assign(const Scalar *values, const std::vector<int> &permutation) {
    // Validate all values before mutating the current numeric generation.
    for (std::size_t i = 0; i < permutation.size(); ++i)
      if (!std::isfinite(values[i]))
        throw std::invalid_argument("Sparse direct values must be finite");
    auto &matrix = *static_cast<Matrix *>(this->get_matrix());
    this->record_numeric_update(permutation.size() * sizeof(Scalar));
    for (std::size_t i = 0; i < permutation.size(); ++i)
      matrix.valuePtr()[permutation[i]] = values[i];
  }
};

template <typename Scalar>
class CpuSparseDirectPlan final : public SparseDirectPlanResource {
 public:
  CpuSparseDirectPlan(Arch arch,
                     Device &device,
                     const SparseDirectPlanDescription &d)
      : SparseDirectPlanResource(d), arch_(arch), device_(device) {
    if (d.implementation != "eigen")
      throw std::invalid_argument("CPU sparse direct requires Eigen");
    const auto *outer =
        static_cast<const int *>(cpu_pointer(device, d.arrays[0]));
    const auto *inner =
        static_cast<const int *>(cpu_pointer(device, d.arrays[1]));
    const auto *values =
        static_cast<const Scalar *>(cpu_pointer(device, d.arrays[2]));
    std::vector<int> offsets(outer, outer + d.rows + 1), indices;
    if (d.nonzeros)
      indices.assign(inner, inner + d.nonzeros);
    validate_native_csr_indices(offsets, indices, d.rows);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(d.nonzeros);
    for (int group = 0; group < int(d.rows); ++group) {
      for (int i = offsets[group]; i < offsets[group + 1]; ++i) {
        if (!std::isfinite(values[i]))
          throw std::invalid_argument("Sparse direct seed values must be finite");
        const auto row = d.format == "csr" ? group : indices[i];
        const auto col = d.format == "csr" ? indices[i] : group;
        triplets.emplace_back(row, col, values[i]);
      }
    }
    using Matrix = Eigen::SparseMatrix<Scalar>;
    Matrix canonical(d.rows, d.rows);
    canonical.setFromTriplets(triplets.begin(), triplets.end());
    canonical.makeCompressed();
    // Freeze the input-to-CSC permutation once. Numeric updates never rebuild
    // triplets, sort entries, or look up coordinates.
    permutation_.resize(d.nonzeros);
    for (std::size_t i = 0; i < triplets.size(); ++i) {
      const auto &entry = triplets[i];
      const auto begin = canonical.outerIndexPtr()[entry.col()];
      const auto end = canonical.outerIndexPtr()[entry.col() + 1];
      permutation_[i] = int(std::lower_bound(canonical.innerIndexPtr() + begin,
                                             canonical.innerIndexPtr() + end,
                                             int(entry.row())) -
                              canonical.innerIndexPtr());
    }
    matrix_ = std::make_unique<OwnedEigenMatrix<Scalar>>(canonical);
    solver_ = make_sparse_solver(d.dtype, d.method, d.ordering);
    if (d.refactor) {
      solver_->analyze_pattern(*matrix_);
    } else if (!solver_->compute(*matrix_)) {
      throw std::invalid_argument("Sparse direct seed factorization failed");
    }
    const auto statistics = matrix_->debug_runtime_statistics();
    known_bytes_ = statistics.operator_owned_reserved_bytes +
                   permutation_.capacity() * sizeof(int);
    // Seed buffers were copied synchronously. Do not retain their addresses.
    description_.arrays = {};
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    // Known owned matrix/permutation only; Eigen factor memory is opaque here.
    return known_bytes_;
  }
  bool retains_input(const std::string &) const override {
    return false;
  }
  void refactor(const Scalar *values) {
    matrix_->assign(values, permutation_);
    solver_->factorize(*matrix_);
    if (!solver_->info())
      throw std::invalid_argument("Sparse direct numeric factorization failed");
  }
  void solve(const void *rhs, void *output) {
    solver_->solve_dense_vector(
        *matrix_, {rhs, description_.dtype, std::size_t(description_.rows)},
        {output, description_.dtype, std::size_t(description_.rows)});
  }

 private:
  Arch arch_;
  Device &device_;
  std::unique_ptr<OwnedEigenMatrix<Scalar>> matrix_;
  std::unique_ptr<SparseSolver> solver_;
  std::vector<int> permutation_;
  std::size_t known_bytes_{0};
};

template <typename Scalar>
class CpuSparseDirectSolve final : public NativeOperation {
 public:
  explicit CpuSparseDirectSolve(const SparseDirectSolveOperation &op)
      : plan_(std::dynamic_pointer_cast<CpuSparseDirectPlan<Scalar>>(op.plan)) {
    if (!plan_)
      throw std::invalid_argument("Sparse direct solve has another plan type");
    if (op.values)
      values_ = static_cast<const Scalar *>(
          cpu_pointer(plan_->device(), *op.values));
    for (const auto &[rhs, output] : op.rhs_outputs)
      pairs_.emplace_back(cpu_pointer(plan_->device(), rhs),
                           cpu_pointer(plan_->device(), output));
  }
  Arch arch() const override {
    return plan_->arch();
  }
  Device &device() const override {
    return plan_->device();
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void run_cpu() override {
    if (plan_->description().refactor)
      plan_->refactor(values_);
    for (const auto &[rhs, output] : pairs_)
      plan_->solve(rhs, output);
  }

 private:
  std::shared_ptr<CpuSparseDirectPlan<Scalar>> plan_;
  const Scalar *values_{nullptr};
  std::vector<std::pair<const void *, void *>> pairs_;
};
#endif
}  // namespace

void register_sparse_direct_operations(NativeOperationRegistry &registry) {
  registry.define<SparseDirectPlanDescription>(validate_plan,
                                               NativeObjectKind::resource);
  registry.register_schema<SparseDirectPlanDescription>(
      "sparse_direct", "plan", {"offsets", "indices", "values"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<SparseDirectPlanDescription> {
        const auto d = decode_plan(p);
        return [d](const NativeOperationBindings &b) {
          auto bound = d;
          bound.arrays = {b.at("offsets").array(), b.at("indices").array(),
                          b.at("values").array()};
          return bound;
        };
      });
  registry.define<SparseDirectSolveOperation>(validate_solve);
  registry.register_schema<SparseDirectSolveOperation>(
      "sparse_direct", "solve", {"plan"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<SparseDirectSolveOperation> {
        operation_json::fields(p, {"rhs_count"});
        const auto count = operation_json::integer(p["rhs_count"]);
        if (!count)
          throw std::invalid_argument("Sparse direct RHS group cannot be empty");
        return [count](const NativeOperationBindings &b) {
          SparseDirectSolveOperation op;
          op.plan = b.at("plan").resource<SparseDirectPlanResource>();
          if (const auto values = b.find("values"); values != b.end())
            op.values = values->second.array();
          for (std::uint64_t i = 0; i < count; ++i)
            op.rhs_outputs.emplace_back(
                b.at("rhs_" + std::to_string(i)).array(),
                b.at("output_" + std::to_string(i)).array());
          return op;
        };
      },
      {}, {"values"},
      {NativeOperationRegistry::BindingGroup{"rhs_count", {"rhs", "output"}}});
#ifdef TI_WITH_LLVM
  registry.register_factory<SparseDirectPlanDescription>(
      host_arch(), NativeOperationMode::cpu,
      [](Arch arch, Device &device, const SparseDirectPlanDescription &d)
          -> std::shared_ptr<NativeResource> {
        if (d.dtype == PrimitiveType::f32)
          return std::make_shared<CpuSparseDirectPlan<float>>(arch, device, d);
        return std::make_shared<CpuSparseDirectPlan<double>>(arch, device, d);
      });
  registry.register_factory<SparseDirectSolveOperation>(
      host_arch(), NativeOperationMode::cpu,
      [](Arch, Device &, const SparseDirectSolveOperation &op)
          -> std::shared_ptr<NativeResource> {
        if (op.plan->description().dtype == PrimitiveType::f32)
          return std::make_shared<CpuSparseDirectSolve<float>>(op);
        return std::make_shared<CpuSparseDirectSolve<double>>(op);
      });
#endif
}
}  // namespace taichi::lang
