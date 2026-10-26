#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/utils.h"

#include <algorithm>
#include <limits>

namespace taichi::lang {

namespace {

using TaskType = OffloadedStmt::TaskType;

/* This pass divides the range-for loop into multithreading blocks, and
 * inserts a new range-for loop that iterates over the number of threads
 * available on the CPU. The computing logics in the original range-for
 * loop are then packed into an inner serial for loop. In a nutshell,
 * the outer offloaded range-for loop is used to parallelize the computation,
 * and inner range-for loop conducts real computation logics.
 *
 * For example, the following code:
 *
 *   @ti.kernel
 *   def foo():
 *     for i in range(1024):
 *       a[i] = i
 *
 * becomes:
 *
 *   @ti.kernel
 *   def foo():
 *     for __chunk in range(2):
 *       block_begin = __chunk * 512
 *       block_end = min(block_begin + 512, 1024)
 *       for i in range(block_begin, block_end):
 *           a[i] = i
 *
 * for a CPU configuration with 8 threads. The existing minimum chunk size
 * is 512. Statically empty trailing chunks need not reach the scheduler.
 *
 * This pass is only applied to range-for loops that are offloaded to
 * CPUs. The number of threads is determined by the config option
 * "cpu_max_num_threads".
 *
 * The effect is that more invarants in the inner most can be identified and
 * moved outside, so that LLVM has more chance to vectorize the innermost
 * loop. This pass especially accelerates simple single level loops, e.g.
 * memcpy and vecadd, even when the loop bounds are determined at runtime.
 */

int static_chunk_count(const OffloadedStmt *task, int num_threads) {
  if (!task->const_begin || !task->const_end || num_threads <= 0) {
    return num_threads;
  }
  // An empty body iteration does not imply an empty auxiliary block. Preserve
  // independently supplied xlogues (normal TLS lowering happens after us).
  for (const auto *block : {task->tls_prologue.get(), task->tls_epilogue.get(),
                            task->bls_prologue.get(), task->bls_epilogue.get()}) {
    if (block && block->size() != 0) {
      return num_threads;
    }
  }
  const int64_t extent =
      int64_t(task->end_value) - int64_t(task->begin_value);
  const int64_t rounded = extent + num_threads - 1;
  // The existing generated partition uses i32 arithmetic. Only trim it when
  // that arithmetic is proven representable; never change the productive
  // partition or infer bounds from the first runtime call.
  const auto lo = std::numeric_limits<int32_t>::min();
  const auto hi = std::numeric_limits<int32_t>::max();
  if (extent < lo || extent > hi || rounded < lo ||
      extent + num_threads > hi) {
    return num_threads;
  }
  const int64_t chunk = std::max<int64_t>(512, rounded / num_threads);
  if (chunk * num_threads > hi ||
      int64_t(task->begin_value) + chunk * num_threads > hi) {
    return num_threads;
  }
  return extent <= 0 ? 0 : int((extent + chunk - 1) / chunk);
}

class MakeCPUMultithreadedRangeFor : public BasicStmtVisitor {
 public:
  explicit MakeCPUMultithreadedRangeFor(const CompileConfig &config)
      : config(config) {
  }

  void visit(Block *block) override {
    for (auto &s_ : block->statements) {
      s_->accept(this);
    }
  }

  void visit(OffloadedStmt *offloaded) override {
    if (offloaded->task_type != TaskType::range_for) {
      return;
    }
    // Device-bounded Graph ranges publish their effective end immediately
    // before the scheduler runs. Keep the original logical range intact so
    // the scheduler can clamp it directly; partitioning a fixed-capacity
    // range into CPU-thread chunks here would make the dynamic end count
    // chunks instead of logical iterations.
    if (offloaded->one_to_one) {
      return;
    }

    const int num_chunks =
        static_chunk_count(offloaded, config.cpu_max_num_threads);
    auto offloaded_body = std::make_unique<Block>();
    auto one = offloaded_body->insert(
        Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 1)));
    auto minimal_block_range = offloaded_body->insert(
        Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 512)));
    auto num_threads = offloaded_body->insert(Stmt::make_typed<ConstStmt>(
        TypedConstant(PrimitiveType::i32, config.cpu_max_num_threads)));
    auto thread_index =
        offloaded_body->insert(Stmt::make_typed<LoopIndexStmt>(offloaded, 0));

    // Retrieve range-for bounds.
    Stmt *begin_stmt;
    Stmt *end_stmt;
    if (offloaded->const_begin) {
      begin_stmt = offloaded_body->insert(Stmt::make_typed<ConstStmt>(
          TypedConstant(PrimitiveType::i32, offloaded->begin_value)));
    } else {
      begin_stmt = offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(
          offloaded->begin_offset, PrimitiveType::i32));
      begin_stmt =
          offloaded_body->insert(Stmt::make<GlobalLoadStmt>(begin_stmt));
    }
    if (offloaded->const_end) {
      end_stmt = offloaded_body->insert(Stmt::make_typed<ConstStmt>(
          TypedConstant(PrimitiveType::i32, offloaded->end_value)));
    } else {
      end_stmt = offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(
          offloaded->end_offset, PrimitiveType::i32));
      end_stmt = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(end_stmt));
    }

    // Inner serial block range is
    // max(((end - begin) + (num_threads - 1)) / num_threads,
    // minimal_block_range)
    auto total_range = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::sub, end_stmt, begin_stmt));
    auto saturated_total_range =
        offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
            BinaryOpType::sub,
            offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
                BinaryOpType::add, total_range, num_threads)),
            one));
    auto block_range = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::floordiv, saturated_total_range, num_threads));
    block_range = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::max, block_range, minimal_block_range));

    // Inner loop begins at
    // begin + block_range * thread_id
    auto block_begin = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::mul, block_range, thread_index));
    block_begin = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::add, begin_stmt, block_begin));

    // Inner loop ends at
    // min(block_begin + block_range), end))
    auto block_end = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::add, block_begin, block_range));
    block_end = offloaded_body->insert(
        Stmt::make_typed<BinaryOpStmt>(BinaryOpType::min, end_stmt, block_end));

    // Create the serial inner loop.
    auto inner_loop = offloaded_body->insert(Stmt::make_typed<RangeForStmt>(
        block_begin, block_end, std::move(offloaded->body),
        /*is_bit_vectorized*/ false, /*num_cpu_threads*/ 1, /*block_dim*/ 1,
        /*strictly_serialized*/ true, offloaded->range_hint));

    irpass::replace_all_usages_with(inner_loop, offloaded, inner_loop);

    // Update the offloaded stmt.
    // Keep the original productive partition and worker request. Only the
    // number of chunks changes when trailing chunks are provably empty.
    // Therefore it has constant begin and end values.
    offloaded->const_begin = true;
    offloaded->const_end = true;
    offloaded->begin_value = 0;
    offloaded->end_value = num_chunks;
    offloaded->body = std::move(offloaded_body);
    offloaded->body->set_parent_stmt(offloaded);
    offloaded->block_dim = 1;
    modified = true;
  }

  static bool run(IRNode *root, const CompileConfig &config) {
    MakeCPUMultithreadedRangeFor pass(config);
    root->accept(&pass);
    return pass.modified;
  }

 private:
  const CompileConfig &config;
  bool modified{false};
};
}  // namespace

namespace irpass {

void make_cpu_multithreaded_range_for(IRNode *root,
                                      const CompileConfig &config) {
  MakeCPUMultithreadedRangeFor::run(root, config);
}

}  // namespace irpass

}  // namespace taichi::lang
