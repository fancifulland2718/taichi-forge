#pragma once

namespace taichi::lang {
class IRNode;
namespace irpass {
// Expand whole-matrix loads, stores and atomics into logical lane accesses.
// Member selection and conversion to physical byte strides are separate.
void scalarize_matrix_ptr_accesses(IRNode *root);
}  // namespace irpass
}  // namespace taichi::lang
