#include "taichi/codegen/cuda/codegen_cuda.h"

#include <vector>
#include <set>
#include <functional>
#include <string_view>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/analysis/gather_snode_tree_dependencies.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/codegen/codegen_utils.h"
#include "taichi/inc/constants.h"

#include "llvm/IR/InlineAsm.h"

namespace taichi::lang {

using namespace llvm;

KernelCodeGenCUDA::KernelCodeGenCUDA(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel *kernel,
    IRNode *ir,
    TaichiLLVMContext &tlctx)
    : KernelCodeGen(compile_config, kernel, ir, tlctx),
      device_caps_(device_caps),
      root_binding_tree_ids_(
          kernel->rets.empty()
              ? irpass::analysis::gather_snode_tree_dependencies(*ir)
              : std::vector<int>{}) {
}

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

static bool is_half2(DataType dt) {
  if (dt->is<TensorType>()) {
    auto tensor_type = dt->as<TensorType>();
    return tensor_type->get_element_type() == PrimitiveType::f16 &&
           tensor_type->get_num_elements() == 2;
  }

  return false;
}

class TaskCodeGenCUDA : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;
  size_t explicit_static_shared_array_bytes{0};

  explicit TaskCodeGenCUDA(int id,
                           const CompileConfig &config,
                           const DeviceCapabilityConfig &device_caps,
                           TaichiLLVMContext &tlctx,
                           const Kernel *kernel,
                           const std::vector<int> &root_binding_tree_ids,
                           IRNode *ir = nullptr)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir, nullptr,
                        &root_binding_tree_ids),
        target_compute_capability_(
            device_caps.contains(DeviceCapability::cuda_compute_capability)
                ? static_cast<int>(device_caps.get(
                      DeviceCapability::cuda_compute_capability))
                : CUDAContext::get_instance().get_compute_capability()) {
  }

  llvm::Value *create_print(std::string tag,
                            DataType dt,
                            llvm::Value *value) override {
    std::string format = data_type_format(dt);
    if (value->getType() == llvm::Type::getFloatTy(*llvm_context)) {
      value =
          builder->CreateFPExt(value, llvm::Type::getDoubleTy(*llvm_context));
    }
    return create_print("[cuda codegen debug] " + tag + " " + format + "\n",
                        {value->getType()}, {value});
  }

  void begin_bls_prologue(OffloadedStmt *stmt) override {
    TI_ASSERT(stmt == current_offload);
    const auto target = cuda::detail::resolve_compute_capability_target(
        target_compute_capability_);
    async_tile_prologue_admission_ =
        cuda::detail::cuda_async_tile_copy_admission(
            target_compute_capability_, target.ptx_version, stmt->bls_size,
            /*copy_bytes=*/4, /*direct_global_to_bls_copy=*/true,
            /*read_only_bls=*/stmt->bls_epilogue == nullptr);
    if (async_tile_prologue_admission_ !=
        cuda::detail::CudaAsyncTileAdmissionReason::kAdmitted) {
      prog->record_cuda_async_tile_candidate(
          async_tile_prologue_admission_);
    }
    in_bls_prologue_ = true;
    emitted_async_tile_copy_ = false;
    async_tile_copy_sites_ = 0;
  }

  void end_bls_prologue(OffloadedStmt *stmt) override {
    TI_ASSERT(stmt == current_offload);
    if (emitted_async_tile_copy_) {
      auto *asm_type = llvm::FunctionType::get(
          llvm::Type::getVoidTy(*llvm_context), /*isVarArg=*/false);
      auto *wait = llvm::InlineAsm::get(asm_type, "cp.async.wait_all;", "",
                                        /*hasSideEffects=*/true);
      builder->CreateCall(wait);
      prog->record_cuda_async_tile_lowering(async_tile_copy_sites_);
    }
    in_bls_prologue_ = false;
  }

  bool emit_async_tile_copy(GlobalStoreStmt *stmt) {
    using AdmissionReason = cuda::detail::CudaAsyncTileAdmissionReason;
    if (!in_bls_prologue_ || !stmt->dest->is<BlockLocalPtrStmt>()) {
      return false;
    }
    const auto reject = [&](AdmissionReason reason) {
      prog->record_cuda_async_tile_candidate(reason);
      return false;
    };
    if (async_tile_prologue_admission_ != AdmissionReason::kAdmitted) {
      return false;
    }
    auto *load = stmt->val->cast<GlobalLoadStmt>();
    if (load == nullptr) {
      return reject(AdmissionReason::kNonDirectAddress);
    }
    auto *source_pointer_type = load->src->ret_type->cast<PointerType>();
    auto *destination_pointer_type =
        stmt->dest->ret_type->cast<PointerType>();
    if (source_pointer_type == nullptr ||
        destination_pointer_type == nullptr ||
        source_pointer_type->is_bit_pointer() ||
        destination_pointer_type->is_bit_pointer() ||
        !stmt->val->ret_type->is<PrimitiveType>()) {
      return reject(AdmissionReason::kNonDirectAddress);
    }
    const int copy_bytes = data_type_size(stmt->val->ret_type);
    const auto target = cuda::detail::resolve_compute_capability_target(
        target_compute_capability_);
    const auto admission = cuda::detail::cuda_async_tile_copy_admission(
        target_compute_capability_, target.ptx_version,
        current_offload->bls_size, copy_bytes,
        /*direct_global_to_bls_copy=*/true,
        /*read_only_bls=*/current_offload->bls_epilogue == nullptr);
    if (admission != AdmissionReason::kAdmitted) {
      return reject(admission);
    }
    prog->record_cuda_async_tile_candidate(AdmissionReason::kAdmitted);

    auto *i64_type = llvm::Type::getInt64Ty(*llvm_context);
    auto *destination =
        builder->CreatePtrToInt(llvm_val[stmt->dest], i64_type);
    auto *source = builder->CreatePtrToInt(llvm_val[load->src], i64_type);
    auto *asm_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(*llvm_context), {i64_type, i64_type},
        /*isVarArg=*/false);
    auto *copy = llvm::InlineAsm::get(
        asm_type,
        fmt::format("{{\n\t.reg .b64 shared_address_64;\n\t"
                    ".reg .b64 global_address_64;\n\t"
                    ".reg .b32 shared_address_32;\n\t"
                    "cvta.to.shared.u64 shared_address_64, $0;\n\t"
                    "cvta.to.global.u64 global_address_64, $1;\n\t"
                    "cvt.u32.u64 shared_address_32, shared_address_64;\n\t"
                    "cp.async.ca.shared.global [shared_address_32], "
                    "[global_address_64], "
                    "{};\n}}",
                    copy_bytes),
        "l,l,~{memory}", /*hasSideEffects=*/true);
    builder->CreateCall(copy, {destination, source});
    emitted_async_tile_copy_ = true;
    ++async_tile_copy_sites_;
    return true;
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (!emit_async_tile_copy(stmt)) {
      TaskCodeGenLLVM::visit(stmt);
    }
  }

  void visit(TexturePtrStmt *stmt) override {
    TI_ERROR_IF(stmt->is_storage,
                "CUDA texture resources are sampled-only; RW texture load "
                "and store are unavailable");
    auto arg_id = stmt->arg_load_stmt->as<ArgLoadStmt>()->arg_id;
    arg_id.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    llvm_val[stmt] = get_struct_arg(arg_id, /*create_load=*/true);
  }

  void visit(TextureOpStmt *stmt) override {
    TI_ERROR_IF(stmt->op != TextureOpType::kSampleLod &&
                    stmt->op != TextureOpType::kFetchTexel,
                "CUDA texture lowering supports only sample_lod and fetch");
    auto *texture_ptr = stmt->texture_ptr->cast<TexturePtrStmt>();
    TI_ASSERT(texture_ptr != nullptr && !texture_ptr->is_storage);
    const int dimensions = texture_ptr->dimensions;
    TI_ASSERT(dimensions >= 1 && dimensions <= 3);
    TI_ASSERT(stmt->args.size() == static_cast<std::size_t>(dimensions + 1));
    const auto texture_arg_id =
        texture_ptr->arg_load_stmt->as<ArgLoadStmt>()->arg_id;
    for (const auto &cached : texture_op_cache_) {
      if (cached.block == builder->GetInsertBlock() &&
          cached.texture_arg_id == texture_arg_id && cached.op == stmt->op &&
          cached.args == stmt->args) {
        llvm_val[stmt] = cached.result;
        return;
      }
    }

    auto *i64_type = llvm::Type::getInt64Ty(*llvm_context);
    auto *f32_type = llvm::Type::getFloatTy(*llvm_context);
    auto *handle = llvm_val[texture_ptr];
    if (handle->getType()->isPointerTy()) {
      handle = builder->CreatePtrToInt(handle, i64_type);
    } else if (handle->getType() != i64_type) {
      handle = builder->CreateZExtOrTrunc(handle, i64_type);
    }

    const bool exact = stmt->op == TextureOpType::kFetchTexel;
    std::vector<llvm::Type *> input_types{i64_type};
    std::vector<llvm::Value *> inputs{handle};
    std::string constraints{"=f,=f,=f,=f,l"};
    std::string coordinates;
    for (int hardware_axis = 0; hardware_axis < dimensions;
         ++hardware_axis) {
      const int logical_axis = dimensions - 1 - hardware_axis;
      auto *coordinate = llvm_val[stmt->args[logical_axis]];
      if (exact) {
        auto dimension_arg_id = texture_arg_id;
        dimension_arg_id.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
        dimension_arg_id.push_back(logical_axis);
        auto *dimension =
            get_struct_arg(dimension_arg_id, /*create_load=*/true);
        coordinate = builder->CreateFDiv(
            builder->CreateFAdd(builder->CreateSIToFP(coordinate, f32_type),
                                llvm::ConstantFP::get(f32_type, 0.5)),
            builder->CreateSIToFP(dimension, f32_type));
      }
      input_types.push_back(coordinate->getType());
      inputs.push_back(coordinate);
      constraints += ",f";
      if (!coordinates.empty()) {
        coordinates += ",";
      }
      coordinates += fmt::format("${}", 5 + hardware_axis);
    }
    if (dimensions == 3) {
      auto *padding = llvm::ConstantFP::get(f32_type, 0.0);
      input_types.push_back(padding->getType());
      inputs.push_back(padding);
      constraints += ",f";
      coordinates += "," + fmt::format("${}", 5 + dimensions);
    }

    const char *geometry =
        dimensions == 1 ? "1d" : dimensions == 2 ? "2d" : "3d";
    std::string assembly =
        "{\n\ttex." + std::string(geometry) + ".v4.f32." +
        "f32 {$0,$1,$2,$3}, [$4, {" + coordinates + "}];\n}";
    auto *result_type = llvm::StructType::get(
        *llvm_context, {f32_type, f32_type, f32_type, f32_type});
    auto *asm_type =
        llvm::FunctionType::get(result_type, input_types, false);
    auto *texture_fetch = llvm::InlineAsm::get(
        asm_type, assembly, constraints, /*hasSideEffects=*/false);
    auto *result = builder->CreateCall(texture_fetch, inputs);

    auto *storage_type = llvm::ArrayType::get(f32_type, 4);
    auto *storage = create_entry_block_alloca(storage_type);
    for (int i = 0; i < 4; ++i) {
      auto *component = builder->CreateExtractValue(result, i);
      auto *destination = builder->CreateGEP(
          storage_type, storage,
          {tlctx->get_constant(0), tlctx->get_constant(i)});
      builder->CreateStore(component, destination);
    }
    llvm_val[stmt] = builder->CreateBitCast(
        storage, llvm::PointerType::get(f32_type, 0));
    texture_op_cache_.push_back({builder->GetInsertBlock(), texture_arg_id,
                                 stmt->op, stmt->args, llvm_val[stmt]});
  }

  void visit(InternalFuncStmt *stmt) override {
    constexpr std::string_view kTextureExtractPrefix{"composite_extract_"};
    if (stmt->func_name.rfind(kTextureExtractPrefix, 0) == 0 &&
        stmt->func_name.size() == kTextureExtractPrefix.size() + 1 &&
        stmt->args.size() == 1 && stmt->args[0]->is<TextureOpStmt>()) {
      const char component_character = stmt->func_name.back();
      TI_ASSERT(component_character >= '0' && component_character <= '3');
      const int component = component_character - '0';
      auto *f32_type = llvm::Type::getFloatTy(*llvm_context);
      auto *component_pointer = builder->CreateGEP(
          f32_type, llvm_val[stmt->args[0]], tlctx->get_constant(component));
      llvm_val[stmt] = builder->CreateLoad(f32_type, component_pointer);
      return;
    }
    TaskCodeGenLLVM::visit(stmt);
  }

  llvm::Value *create_print(const std::string &format,
                            const std::vector<llvm::Type *> &types,
                            const std::vector<llvm::Value *> &values) {
    auto stype = llvm::StructType::get(*llvm_context, types, false);
    auto value_arr = builder->CreateAlloca(stype);
    for (int i = 0; i < values.size(); i++) {
      auto value_ptr = builder->CreateGEP(
          stype, value_arr, {tlctx->get_constant(0), tlctx->get_constant(i)});
      builder->CreateStore(values[i], value_ptr);
    }
    return LLVMModuleBuilder::call(
        builder.get(), "vprintf",
        create_global_string(format, "format_string"),
        builder->CreateBitCast(value_arr,
                               llvm::PointerType::get(*llvm_context, 0)));
  }

  std::tuple<llvm::Value *, llvm::Type *> create_value_and_type(
      llvm::Value *value,
      DataType dt) {
    auto value_type = tlctx->get_data_type(dt);
    if (dt->is_primitive(PrimitiveTypeID::f32) ||
        dt->is_primitive(PrimitiveTypeID::f16)) {
      value_type = tlctx->get_data_type(PrimitiveType::f64);
      value = builder->CreateFPExt(value, value_type);
    }
    if (dt->is_primitive(PrimitiveTypeID::i8)) {
      value_type = tlctx->get_data_type(PrimitiveType::i16);
      value = builder->CreateSExt(value, value_type);
    }
    if (dt->is_primitive(PrimitiveTypeID::u8)) {
      value_type = tlctx->get_data_type(PrimitiveType::u16);
      value = builder->CreateZExt(value, value_type);
    }
    if (dt->is_primitive(PrimitiveTypeID::u1)) {
      value_type = tlctx->get_data_type(PrimitiveType::i32);
      value = builder->CreateZExt(value, value_type);
    }
    return std::make_tuple(value, value_type);
  }

  void visit(PrintStmt *stmt) override {
    TI_ASSERT_INFO(stmt->contents.size() < 32,
                   "CUDA `print()` doesn't support more than 32 entries");

    std::vector<llvm::Type *> types;
    std::vector<llvm::Value *> values;

    std::string formats;
    size_t num_contents = 0;
    for (auto i = 0; i < stmt->contents.size(); ++i) {
      auto const &content = stmt->contents[i];
      auto const &format = stmt->formats[i];

      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);

        auto &&merged_format = merge_printf_specifier(
            format, data_type_format(arg_stmt->ret_type));
        // CUDA supports all conversions, but not 'F'.
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#format-specifiers
        std::replace(merged_format.begin(), merged_format.end(), 'F', 'f');
        formats += merged_format;

        auto value = llvm_val[arg_stmt];
        auto value_type = value->getType();
        if (arg_stmt->ret_type->is<TensorType>()) {
          auto dtype = arg_stmt->ret_type->cast<TensorType>();
          num_contents += dtype->get_num_elements();
          auto elem_type = dtype->get_element_type();
          for (int i = 0; i < dtype->get_num_elements(); ++i) {
            llvm::Value *elem_value;
            if (codegen_vector_type(compile_config)) {
              TI_ASSERT(llvm::dyn_cast<llvm::VectorType>(value_type));
              elem_value = builder->CreateExtractElement(value, i);
            } else {
              TI_ASSERT(llvm::dyn_cast<llvm::ArrayType>(value_type));
              elem_value = builder->CreateExtractValue(value, i);
            }
            auto [casted_value, elem_value_type] =
                create_value_and_type(elem_value, elem_type);
            types.push_back(elem_value_type);
            values.push_back(casted_value);
          }
        } else {
          num_contents++;
          auto [val, dtype] = create_value_and_type(value, arg_stmt->ret_type);
          types.push_back(dtype);
          values.push_back(val);
        }
      } else {
        num_contents += 1;
        auto arg_str = std::get<std::string>(content);

        auto value = create_global_string(arg_str, "content_string");
        auto char_type =
            llvm::Type::getInt8Ty(*tlctx->get_this_thread_context());
        auto value_type = llvm::PointerType::get(char_type, 0);

        types.push_back(value_type);
        values.push_back(value);
        formats += "%s";
      }
      TI_ASSERT_INFO(num_contents < 32,
                     "CUDA `print()` doesn't support more than 32 entries");
    }

    llvm_val[stmt] = create_print(formats, types, values);
  }

  void visit(AllocaStmt *stmt) override {
    // Override shared memory codegen logic for large shared memory
    auto tensor_type = stmt->ret_type.ptr_removed()->cast<TensorType>();
    if (tensor_type && stmt->is_shared) {
      size_t shared_array_bytes =
          tensor_type->get_num_elements() *
          data_type_size(tensor_type->get_element_type());
      constexpr size_t kSharedArrayAlignment = 8;
      explicit_static_shared_array_bytes =
          (explicit_static_shared_array_bytes + kSharedArrayAlignment - 1) /
          kSharedArrayAlignment * kSharedArrayAlignment;
      explicit_static_shared_array_bytes += shared_array_bytes;
      const size_t task_shared_array_bytes =
          explicit_static_shared_array_bytes +
          (current_offload == nullptr ? 0 : current_offload->bls_size);
      TI_ERROR_IF(
          task_shared_array_bytes > cuda_shared_array_limit_bytes,
          "CUDA task requests {} aggregate bytes of static shared memory, "
          "exceeding the supported 48 KiB "
          "per-block limit. CUDA opt-in dynamic shared memory is disabled "
          "because larger allocations can trigger "
          "CUDA_ERROR_ILLEGAL_ADDRESS (IMA), including during Graph replay.",
          task_shared_array_bytes);
      // Keep the interned TensorType immutable. The former dynamic-shared
      // lowering changed its shape to {0} in place, corrupting later kernels
      // that reused the same type after compilation or ti.reset().
      auto *shared_array_type = tlctx->get_data_type(tensor_type);
      auto base = new llvm::GlobalVariable(
          *module, shared_array_type, false,
          llvm::GlobalValue::ExternalLinkage, nullptr,
          fmt::format("shared_array_{}", stmt->id), nullptr,
          llvm::GlobalVariable::NotThreadLocal, 3 /*addrspace=shared*/);
      base->setAlignment(llvm::MaybeAlign(8));
      auto ptr_type = llvm::PointerType::get(shared_array_type, 0);
      llvm_val[stmt] = builder->CreatePointerCast(base, ptr_type);
    } else {
      TaskCodeGenLLVM::visit(stmt);
    }
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    // functions from libdevice
    auto input = llvm_val[stmt->operand];
    auto input_taichi_type = stmt->operand->ret_type;
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {
      // Promote to f32 since we don't have f16 support for extra unary ops in
      // libdevice.
      input =
          builder->CreateFPExt(input, llvm::Type::getFloatTy(*llvm_context));
      input_taichi_type = PrimitiveType::f32;
    }

    auto op = stmt->op_type;

#define UNARY_STD(x)                                                    \
  else if (op == UnaryOpType::x) {                                      \
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {        \
      llvm_val[stmt] = call("__nv_" #x "f", input);                     \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = call("__nv_" #x, input);                         \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) { \
      llvm_val[stmt] = call(#x, input);                                 \
    } else {                                                            \
      TI_NOT_IMPLEMENTED                                                \
    }                                                                   \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_fabsf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_fabs", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("__nv_abs", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i64)) {
        llvm_val[stmt] = call("__nv_llabs", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_sqrtf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_sqrt", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::frexp) {
      auto stype = tlctx->get_data_type(stmt->ret_type.ptr_removed());
      auto res = builder->CreateAlloca(stype);
      auto frac_ptr = builder->CreateStructGEP(stype, res, 0);
      auto exp_ptr = builder->CreateStructGEP(stype, res, 1);
      // __nv_frexp onlys takes in double
      auto double_input =
          input_taichi_type->is_primitive(PrimitiveTypeID::f32)
              ? builder->CreateFPExt(
                    input,
                    llvm::Type::getDoubleTy(*tlctx->get_this_thread_context()))
              : input;
      auto frac = call("__nv_frexp", double_input, exp_ptr);
      auto output =
          input_taichi_type->is_primitive(PrimitiveTypeID::f32)
              ? builder->CreateFPTrunc(
                    frac,
                    llvm::Type::getFloatTy(*tlctx->get_this_thread_context()))
              : frac;
      builder->CreateStore(output, frac_ptr);
      llvm_val[stmt] = res;
    } else if (op == UnaryOpType::popcnt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::u64) ||
          input_taichi_type->is_primitive(PrimitiveTypeID::i64)) {
        stmt->ret_type = PrimitiveType::i32;
        llvm_val[stmt] = call("__nv_popcll", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32) ||
                 input_taichi_type->is_primitive(PrimitiveTypeID::u32)) {
        llvm_val[stmt] = call("__nv_popc", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::clz) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        stmt->ret_type = PrimitiveType::i32;
        llvm_val[stmt] = call("__nv_clz", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i64)) {
        llvm_val[stmt] = call("__nv_clzll", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::log) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        // logf has fast-math option
        llvm_val[stmt] = call(
            compile_config.fast_math ? "__nv_fast_logf" : "__nv_logf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_log", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("log", input);
      } else {
        TI_ERROR("log() for type {} is not supported",
                 input_taichi_type.to_string());
      }
    } else if (op == UnaryOpType::sin) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        // sinf has fast-math option
        llvm_val[stmt] = call(
            compile_config.fast_math ? "__nv_fast_sinf" : "__nv_sinf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_sin", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("sin", input);
      } else {
        TI_ERROR("sin() for type {} is not supported",
                 input_taichi_type.to_string());
      }
    } else if (op == UnaryOpType::cos) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        // cosf has fast-math option
        llvm_val[stmt] = call(
            compile_config.fast_math ? "__nv_fast_cosf" : "__nv_cosf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_cos", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("cos", input);
      } else {
        TI_ERROR("cos() for type {} is not supported",
                 input_taichi_type.to_string());
      }
    }
    UNARY_STD(exp)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(sgn)
    UNARY_STD(acos)
    UNARY_STD(asin)
    else {
      TI_P(unary_op_type_name(op));
      TI_NOT_IMPLEMENTED
    }
#undef UNARY_STD
    if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      // Convert back to f16.
      llvm_val[stmt] = builder->CreateFPTrunc(
          llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
    }
  }

  // Not all reduction statements can be optimized.
  // If the operation cannot be optimized, this function returns nullptr.
  llvm::Value *optimized_reduction(AtomicOpStmt *stmt) override {
    if (!stmt->is_reduction) {
      return nullptr;
    }
    TI_ASSERT(stmt->val->ret_type->is<PrimitiveType>());
    PrimitiveTypeID prim_type =
        stmt->val->ret_type->cast<PrimitiveType>()->type;

    std::unordered_map<PrimitiveTypeID,
                       std::unordered_map<AtomicOpType, std::string>>
        fast_reductions;

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::add] = "reduce_add_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::add] = "reduce_add_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::min] = "reduce_min_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::min] = "reduce_min_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::max] = "reduce_max_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::max] = "reduce_max_f32";

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_and] =
        "reduce_and_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_or] =
        "reduce_or_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_xor] =
        "reduce_xor_i32";

    AtomicOpType op = stmt->op_type;
    if (fast_reductions.find(prim_type) == fast_reductions.end()) {
      return nullptr;
    }
    TI_ASSERT(fast_reductions.at(prim_type).find(op) !=
              fast_reductions.at(prim_type).end());
    return call(fast_reductions.at(prim_type).at(op), llvm_val[stmt->dest],
                llvm_val[stmt->val]);
  }

  void visit(AtomicOpStmt *atomic_stmt) override {
    auto dest_type = atomic_stmt->dest->ret_type.ptr_removed();
    auto val_type = atomic_stmt->val->ret_type;

    // Half2 atomic_add is supported starting from sm_60. CUDA AOT passes an
    // explicit target capability; ordinary JIT falls back to the current
    // device capability.

    std::string cuda_library_path = get_custom_cuda_library_path();
    if (is_half2(dest_type) && is_half2(val_type) &&
        atomic_stmt->op_type == AtomicOpType::add &&
        target_compute_capability_ >= 60 &&
        !cuda_library_path.empty()) {
      /*
        Half2 optimization for float16 atomic add

        [CHI IR]
            TensorType<2 x f16> old_val = atomic_add(TensorType<2 x f16>
        dest_ptr*, TensorType<2 x f16> val)

        [CodeGen]
            old_val_ptr = Alloca(TensorType<2 x f16>)

            val_ptr = Alloca(TensorType<2 x f16>)
            GEP(val_ptr, 0) = ExtractValue(val, 0)
            GEP(val_ptr, 1) = ExtractValue(val, 1)

            half2_atomic_add(dest_ptr, old_val_ptr, val_ptr)

            old_val = Load(old_val_ptr)
      */
      // Allocate old_val_ptr to store the result of atomic_add
      auto char_type = llvm::Type::getInt8Ty(*tlctx->get_this_thread_context());
      auto half_type = llvm::Type::getHalfTy(*tlctx->get_this_thread_context());
      auto ptr_type = llvm::PointerType::get(char_type, 0);

      llvm::Value *old_val = builder->CreateAlloca(half_type);
      llvm::Value *old_val_ptr = builder->CreateBitCast(old_val, ptr_type);

      // Prepare dest_ptr via pointer cast
      llvm::Value *dest_half2_ptr =
          builder->CreateBitCast(llvm_val[atomic_stmt->dest], ptr_type);

      // Prepare value_ptr from val
      llvm::ArrayType *array_type = llvm::ArrayType::get(half_type, 2);
      llvm::Value *value_ptr = builder->CreateAlloca(array_type);
      llvm::Value *value_ptr0 =
          builder->CreateGEP(array_type, value_ptr,
                             {tlctx->get_constant(0), tlctx->get_constant(0)});
      llvm::Value *value_ptr1 =
          builder->CreateGEP(array_type, value_ptr,
                             {tlctx->get_constant(0), tlctx->get_constant(1)});
      llvm::Value *value0 =
          builder->CreateExtractValue(llvm_val[atomic_stmt->val], {0});
      llvm::Value *value1 =
          builder->CreateExtractValue(llvm_val[atomic_stmt->val], {1});
      builder->CreateStore(value0, value_ptr0);
      builder->CreateStore(value1, value_ptr1);
      llvm::Value *value_half2_ptr =
          builder->CreateBitCast(value_ptr, ptr_type);
      // Defined in taichi/runtime/llvm/runtime_module/cuda_runtime.cu
      call("half2_atomic_add", dest_half2_ptr, old_val_ptr, value_half2_ptr);

      llvm_val[atomic_stmt] = builder->CreateLoad(half_type, old_val);
      return;
    }

    TaskCodeGenLLVM::visit(atomic_stmt);
  }

  void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_xlogue(stmt->tls_prologue);

    llvm::Value *bls_prologue = nullptr;
    if (stmt->bls_prologue) {
      auto guard = get_function_creation_guard(get_xlogue_argument_types());
      begin_bls_prologue(stmt);
      stmt->bls_prologue->accept(this);
      end_bls_prologue(stmt);
      bls_prologue = guard.body;
    }

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           get_tls_buffer_type(), tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    auto epilogue = create_xlogue(stmt->tls_epilogue);

    auto [begin, end] = get_range_for_bounds(stmt);
    if (stmt->external_shared_staged) {
      TI_ASSERT(bls_prologue != nullptr);
      if (stmt->external_shared_iteration_shape.size() == 2) {
        TI_ASSERT(stmt->external_shared_tile_shape.size() == 2);
        call("gpu_parallel_range_for_shared_staged_2d", get_arg(0), begin,
             end,
             tlctx->get_constant(stmt->external_shared_iteration_shape[0]),
             tlctx->get_constant(stmt->external_shared_iteration_shape[1]),
             tlctx->get_constant(stmt->external_shared_tile_shape[0]),
             tlctx->get_constant(stmt->external_shared_tile_shape[1]),
             tls_prologue, bls_prologue, body, epilogue,
             tlctx->get_constant(stmt->tls_size));
      } else {
        call("gpu_parallel_range_for_shared_staged", get_arg(0), begin, end,
             tls_prologue, bls_prologue, body, epilogue,
             tlctx->get_constant(stmt->tls_size));
      }
    } else if (stmt->one_to_one) {
      // CUDA bounded Graph payloads keep the backend's ordinary
      // saturation-capped grid-stride scheduler. Only the logical range end
      // is loaded from the device extent; CUDA 12.4 node updates may trim the
      // physical grid further, but correctness never depends on that update.
      auto *i64_type = llvm::Type::getInt64Ty(*llvm_context);
      auto *begin_i64 = builder->CreateSExt(begin, i64_type);
      auto *end_i64 = builder->CreateSExt(end, i64_type);
      auto *count_i64 = builder->CreateSExt(
          load_cuda_bounded_extent_count(), i64_type);
      auto *bounded_end_i64 = builder->CreateAdd(begin_i64, count_i64);
      bounded_end_i64 = builder->CreateSelect(
          builder->CreateICmpSLT(bounded_end_i64, end_i64), bounded_end_i64,
          end_i64);
      auto *bounded_end =
          builder->CreateTrunc(bounded_end_i64, builder->getInt32Ty());
      call("gpu_parallel_range_for", get_arg(0), begin, bounded_end,
           tls_prologue, body, epilogue,
           tlctx->get_constant(stmt->tls_size));
    } else {
      call("gpu_parallel_range_for", get_arg(0), begin, end, tls_prologue,
           body, epilogue, tlctx->get_constant(stmt->tls_size));
    }
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_mesh_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           get_tls_buffer_type(), tlctx->get_data_type<int>()});

      for (int i = 0; i < stmt->mesh_prologue->size(); i++) {
        auto &s = stmt->mesh_prologue->statements[i];
        s->accept(this);
      }

      if (stmt->bls_prologue) {
        stmt->bls_prologue->accept(this);
        call("block_barrier");  // "__syncthreads()"
      }

      auto loop_test_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_test", func);
      auto loop_body_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_body", func);
      auto func_exit =
          llvm::BasicBlock::Create(*llvm_context, "func_exit", func);
      auto i32_ty = llvm::Type::getInt32Ty(*llvm_context);
      auto loop_index = create_entry_block_alloca(i32_ty);
      llvm::Value *thread_idx =
          builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
      llvm::Value *block_dim = builder->CreateIntrinsic(
          Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});
      builder->CreateStore(thread_idx, loop_index);
      builder->CreateBr(loop_test_bb);

      {
        builder->SetInsertPoint(loop_test_bb);
        auto cond = builder->CreateICmp(
            llvm::CmpInst::Predicate::ICMP_SLT,
            builder->CreateLoad(i32_ty, loop_index),
            llvm_val[stmt->owned_num_local.find(stmt->major_from_type)
                         ->second]);
        builder->CreateCondBr(cond, loop_body_bb, func_exit);
      }

      {
        builder->SetInsertPoint(loop_body_bb);
        loop_vars_llvm[stmt].push_back(loop_index);
        for (int i = 0; i < stmt->body->size(); i++) {
          auto &s = stmt->body->statements[i];
          s->accept(this);
        }
        builder->CreateStore(
            builder->CreateAdd(builder->CreateLoad(i32_ty, loop_index),
                               block_dim),
            loop_index);
        builder->CreateBr(loop_test_bb);
        builder->SetInsertPoint(func_exit);
      }

      if (stmt->bls_epilogue) {
        call("block_barrier");  // "__syncthreads()"
        stmt->bls_epilogue->accept(this);
      }

      body = guard.body;
    }

    auto tls_epilogue = create_mesh_xlogue(stmt->tls_epilogue);

    call("gpu_parallel_mesh_for", get_arg(0),
         tlctx->get_constant(stmt->mesh->num_patches), tls_prologue, body,
         tls_epilogue, tlctx->get_constant(stmt->tls_size));
  }

  void emit_cuda_gc(OffloadedStmt *stmt) {
    if (compile_config.cuda_pointer_fast_reset &&
        compile_config.cuda_pointer_deterministic_slot &&
        stmt->snode->type == SNodeType::pointer) {
      int64 total_from_root = 1;
      for (int j = 0; j < taichi_max_num_indices; j++) {
        total_from_root *= stmt->snode->extractors[j].num_elements_from_root;
      }
      if (stmt->snode->num_cells_per_container == total_from_root) {
        return;
      }
    }
    const uint64 runtime_key =
        (uint64(static_cast<uint32>(stmt->snode->get_snode_tree_id())) << 32) |
        uint64(static_cast<uint32>(stmt->snode->runtime_local_id));
    auto snode_runtime_key = tlctx->get_constant(runtime_key);
    {
      init_offloaded_task_function(stmt, "gather_list");
      call("gc_parallel_0", get_context(), snode_runtime_key);
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "reinit_lists");
      call("gc_parallel_1", get_context(), snode_runtime_key);
      finalize_offloaded_task_function();
      current_task->grid_dim = 1;
      current_task->block_dim = 1;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "zero_fill");
      call("gc_parallel_2", get_context(), snode_runtime_key);
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
  }

  bool kernel_argument_by_val() const override {
    return true;  // on CUDA, pass the argument by value
  }

  llvm::Value *create_intrinsic_load(llvm::Value *ptr,
                                     llvm::Type *ty) override {
    // Issue an "__ldg" instruction to cache data in the read-only data cache.
    //
    // LLVM 20 removed the ``llvm::Intrinsic::nvvm_ldg_global_{f,i}``
    // intrinsics. The recommended replacement is an ordinary load tagged
    // with ``!invariant.load`` metadata: the NVPTX backend lowers such
    // loads to ``ld.global.nc`` (== ``__ldg``) just like the old intrinsics
    // did. This form is also accepted by LLVM 19, so we use it
    // unconditionally.
    //
    // Special treatment for bool types. As the underlying ld.global.nc does
    // not support 1-bit integer, so we convert them to i8 first.
    llvm::Type *load_ty = ty;
    llvm::Value *load_ptr = ptr;
    const bool is_bool = ty->getScalarSizeInBits() == 1;
    if (is_bool) {
      load_ty = tlctx->get_data_type<uint8>();
      load_ptr = builder->CreatePointerCast(
          ptr, llvm::PointerType::get(load_ty, 0));
    }
    auto *load = builder->CreateLoad(load_ty, load_ptr);
    load->setMetadata(
        llvm::LLVMContext::MD_invariant_load,
        llvm::MDNode::get(*llvm_context, llvm::ArrayRef<llvm::Metadata *>{}));
    if (is_bool) {
      return builder->CreateIsNotNull(load);
    }
    return load;
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (auto get_ch = stmt->src->cast<GetChStmt>()) {
      bool should_cache_as_read_only = current_offload->mem_access_opt.has_flag(
          get_ch->output_snode, SNodeAccessFlag::read_only);
      create_global_load(stmt, should_cache_as_read_only);
    } else {
      create_global_load(stmt, false);
    }
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                     stmt->bls_size);
    bls_buffer = new GlobalVariable(
        *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "bls_buffer", nullptr, llvm::GlobalVariable::NotThreadLocal,
        3 /*addrspace=shared*/);
    bls_buffer->setAlignment(llvm::MaybeAlign(8));
  }

  void visit(OffloadedStmt *stmt) override {
    explicit_static_shared_array_bytes = 0;
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
#if defined(TI_WITH_CUDA)
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::gc) {
      // gc has 3 kernels, so we treat it specially
      emit_cuda_gc(stmt);
    } else {
      init_offloaded_task_function(stmt);
      if (stmt->task_type == Type::serial) {
        stmt->body->accept(this);
      } else if (stmt->task_type == Type::range_for) {
        create_offload_range_for(stmt);
      } else if (stmt->task_type == Type::struct_for) {
        create_offload_struct_for(stmt);
      } else if (stmt->task_type == Type::mesh_for) {
        create_offload_mesh_for(stmt);
      } else if (stmt->task_type == Type::listgen) {
        emit_list_gen(stmt);
      } else {
        TI_NOT_IMPLEMENTED
      }
      finalize_offloaded_task_function();
      current_task->grid_dim = stmt->grid_dim;
      if (stmt->task_type == Type::range_for &&
          !(stmt->external_shared_staged &&
            stmt->external_shared_iteration_shape.size() == 2)) {
        if (stmt->const_begin && stmt->const_end) {
          int num_threads = stmt->end_value - stmt->begin_value;
          int grid_dim = ((num_threads % stmt->block_dim) == 0)
                             ? (num_threads / stmt->block_dim)
                             : (num_threads / stmt->block_dim) + 1;
          grid_dim = std::max(grid_dim, 1);
          current_task->grid_dim = std::min(stmt->grid_dim, grid_dim);
        }
      }
      if (stmt->task_type == Type::listgen) {
        int query_max_block_per_sm;
        CUDADriver::get_instance().device_get_attribute(
            &query_max_block_per_sm,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
        int num_SMs;
        CUDADriver::get_instance().device_get_attribute(
            &num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
        const int saturating = num_SMs * query_max_block_per_sm;
        int chosen = saturating;
        size_t parent_list_max = 1;
        if (stmt->snode != nullptr) {
          for (auto *sn = stmt->snode->parent;
               sn != nullptr && sn->type != SNodeType::root;
               sn = sn->parent) {
            parent_list_max *= (size_t)sn->num_cells_per_container;
            if (parent_list_max >= (size_t)saturating) {
              parent_list_max = (size_t)saturating;
              break;
            }
          }
          current_task->sparse_list_parent_grid_bound =
              std::max(1, std::min(saturating, (int)parent_list_max));
        }
        // §16.13 (S3, 2026-05-05): when the opt-in flag is set, derive a
        // static upper bound on the number of parent_list elements (i.e.
        // the i-axis count of element_listgen_nonroot's outer loop, see
        // taichi/runtime/llvm/runtime_module/runtime.cpp) and cap the
        // grid_dim to it. The bound is the product of
        // num_cells_per_container of every strict ancestor of stmt->snode
        // (root excluded). Underestimating is safe because the runtime
        // loop is grid-stride: any extra parent elements get serviced by
        // existing blocks. On a typical sparse tree
        // `root.bitmasked(64).f32`, parent_list has exactly 1 entry, so
        // saturating to thousands of blocks wastes >99% of launched
        // threads on dispatch overhead and atomic contention.
        if (compile_config.listgen_static_grid_dim &&
            current_task->sparse_list_parent_grid_bound > 0) {
          chosen = current_task->sparse_list_parent_grid_bound;
        }
        current_task->grid_dim = chosen;
      }
      current_task->block_dim = stmt->block_dim;
      annotate_current_task_metadata(stmt);
      current_task->static_shared_array_bytes = static_cast<int>(
          stmt->bls_size + explicit_static_shared_array_bytes);
      current_task->dynamic_shared_array_bytes = 0;
      TI_ASSERT(current_task->grid_dim != 0);
      TI_ASSERT(current_task->block_dim != 0);
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    current_offload = nullptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      TaskCodeGenLLVM::visit_call_bitcode(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    auto op = stmt->op_type;
    if (op != BinaryOpType::atan2 && op != BinaryOpType::pow) {
      return TaskCodeGenLLVM::visit(stmt);
    }

    auto ret_type = stmt->ret_type;

    llvm::Value *lhs = llvm_val[stmt->lhs];
    llvm::Value *rhs = llvm_val[stmt->rhs];

    // This branch contains atan2 and pow which use runtime.cpp function for
    // **real** type. We don't have f16 support there so promoting to f32 is
    // necessary.
    if (stmt->lhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      lhs = builder->CreateFPExt(lhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (stmt->rhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      rhs = builder->CreateFPExt(rhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (ret_type->is_primitive(PrimitiveTypeID::f16)) {
      ret_type = PrimitiveType::f32;
    }

    if (op == BinaryOpType::atan2) {
      if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_atan2f", lhs, rhs);
      } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_atan2", lhs, rhs);
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else {
      // Note that ret_type here cannot be integral because pow with an
      // integral exponent has been demoted in the demote_operations pass
      if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_powf", lhs, rhs);
      } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_pow", lhs, rhs);
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    }

    // Convert back to f16 if applicable.
    if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      llvm_val[stmt] = builder->CreateFPTrunc(
          llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
    }
  }

 private:
  struct CachedTextureOp {
    llvm::BasicBlock *block;
    std::vector<int> texture_arg_id;
    TextureOpType op;
    std::vector<Stmt *> args;
    llvm::Value *result;
  };

  std::vector<CachedTextureOp> texture_op_cache_;

  // CudaBoundedRangeBinding is a private Graph argument prefix whose layout is
  // asserted in taichi/program/context.h. Keep the byte offsets explicit here
  // so the device module does not need a new split-runtime type or symbol.
  static constexpr int64 kCudaBoundedRangeBindingSize = 16;

  llvm::Value *load_cuda_bounded_extent_count() {
    auto *i8_type = llvm::Type::getInt8Ty(*llvm_context);
    auto *i32_type = builder->getInt32Ty();
    auto *i64_type = llvm::Type::getInt64Ty(*llvm_context);
    auto *pointer_type = llvm::PointerType::get(*llvm_context, 0);
    auto *runtime_context_type = get_runtime_type("RuntimeContext");
    auto *zero = tlctx->get_constant(0);
    auto *arg_buffer_field = builder->CreateGEP(
        runtime_context_type, get_context(), {zero, zero});
    auto *arg_buffer = builder->CreateAlignedLoad(
        pointer_type, arg_buffer_field, llvm::Align(8), "bounded_args");
    auto *binding = builder->CreateInBoundsGEP(
        i8_type, arg_buffer,
        llvm::ConstantInt::getSigned(i64_type,
                                     -kCudaBoundedRangeBindingSize),
        "cuda_bounded_binding");
    auto *extent_bits = builder->CreateAlignedLoad(
        i64_type, binding, llvm::Align(8), "cuda_bounded_extent_address");
    auto *capacity_address = builder->CreateInBoundsGEP(
        i8_type, binding, llvm::ConstantInt::get(i64_type, 8));
    auto *capacity = builder->CreateAlignedLoad(
        i32_type, capacity_address, llvm::Align(4), "cuda_bounded_capacity");
    auto *extent = builder->CreateIntToPtr(extent_bits, pointer_type);
    auto *raw_count = builder->CreateAlignedLoad(
        i32_type, extent, llvm::Align(4), "cuda_bounded_raw_count");
    auto *negative = builder->CreateICmpSLT(raw_count, zero);
    auto *above_capacity = builder->CreateICmpSGT(raw_count, capacity);
    auto *nonnegative = builder->CreateSelect(negative, zero, raw_count);
    auto *count =
        builder->CreateSelect(above_capacity, capacity, nonnegative);

    // Exactly one physical lane normalizes invalid producer output and sets
    // sticky overflow. All lanes use the locally clamped value, so there is no
    // grid-wide synchronization requirement before entering the range loop.
    auto *thread_idx =
        builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
    auto *block_idx = builder->CreateIntrinsic(
        Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, {});
    auto *leader = builder->CreateAnd(builder->CreateICmpEQ(thread_idx, zero),
                                      builder->CreateICmpEQ(block_idx, zero));
    auto *invalid = builder->CreateOr(negative, above_capacity);
    auto *normalize = llvm::BasicBlock::Create(
        *llvm_context, "cuda_bounded_normalize", func);
    auto *normalized = llvm::BasicBlock::Create(
        *llvm_context, "cuda_bounded_normalized", func);
    builder->CreateCondBr(builder->CreateAnd(leader, invalid), normalize,
                          normalized);
    builder->SetInsertPoint(normalize);
    builder->CreateAlignedStore(count, extent, llvm::Align(4));
    auto *overflow = builder->CreateInBoundsGEP(
        i32_type, extent, llvm::ConstantInt::get(i64_type, 1));
    builder->CreateAlignedStore(tlctx->get_constant(1), overflow,
                                llvm::Align(4));
    builder->CreateBr(normalized);
    builder->SetInsertPoint(normalized);
    return count;
  }

  int target_compute_capability_{0};
  bool in_bls_prologue_{false};
  cuda::detail::CudaAsyncTileAdmissionReason
      async_tile_prologue_admission_{
          cuda::detail::CudaAsyncTileAdmissionReason::kCostGate};
  bool emitted_async_tile_copy_{false};
  std::size_t async_tile_copy_sites_{0};

  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx =
        builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
    auto block_dim =
        builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});
    return std::make_tuple(thread_idx, block_dim);
  }
};

LLVMCompiledTask KernelCodeGenCUDA::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenCUDA gen(task_codegen_id, config, device_caps_,
                      get_taichi_llvm_context(), kernel,
                      root_binding_tree_ids_, block);
  return gen.run_compilation();
}

}  // namespace taichi::lang
