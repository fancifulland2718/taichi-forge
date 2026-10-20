#pragma once

#include "gtest/gtest.h"
#include "taichi/aot/module_loader.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang::aot_test_utils {
template <typename Complete>
void check_value_parameters(aot::Module *module, Complete complete) {
  auto *kernel = module->get_kernel("value_parameters");
  ASSERT_NE(kernel, nullptr);
  ASSERT_TRUE(kernel->value_parameter_types.has_value());
  ASSERT_EQ(kernel->value_parameter_types->size(), 2);
  auto vector = kernel->value_parameter_types->at({0});
  ASSERT_TRUE(vector->is<TensorType>());
  EXPECT_EQ(vector->as<TensorType>()->get_shape(), std::vector<int>({3}));
  auto record = kernel->value_parameter_types->at({1});
  ASSERT_TRUE(record->is<StructType>());
  EXPECT_EQ(record->as<StructType>()->get_element_type({3})->as<TensorType>()->get_shape(),
            std::vector<int>({3, 1}));
  auto *alias = module->get_kernel("value_parameters_alias");
  ASSERT_NE(alias, nullptr);
  EXPECT_EQ(alias->value_parameter_types, kernel->value_parameter_types);
  for (auto *entry : {kernel, alias}) {
    LaunchContextBuilder arguments(entry);
    for (int i = 0; i < 3; ++i) {
      arguments.set_arg_float({0, i}, i + 1);
      arguments.set_arg_float({1, 1, i}, i + 2);
      arguments.set_arg_float({1, 3, i}, i + 5);
    }
    arguments.set_arg_int({1, 0}, 7);
    for (int i = 0; i < 4; ++i)
      arguments.set_arg_float({1, 2, i}, i + 1);
    entry->launch(arguments);
    complete();
    EXPECT_FLOAT_EQ(arguments.get_ret<float>(0), 67.0f);
  }
  auto *wide = module->get_kernel("wide_parameters");
  ASSERT_NE(wide, nullptr);
  ASSERT_TRUE(wide->value_parameter_types.has_value());
  EXPECT_EQ(wide->value_parameter_types->at({0})->as<TensorType>()->get_shape(),
            std::vector<int>({65}));
  EXPECT_EQ(wide->value_parameter_types->at({1})->as<TensorType>()->get_shape(),
            std::vector<int>({9, 9}));
  LaunchContextBuilder arguments(wide);
  for (int i = 0; i < 65; ++i)
    arguments.set_arg_float({0, i}, i + 1);
  for (int i = 0; i < 81; ++i)
    arguments.set_arg_float({1, i}, i + 1);
  wide->launch(arguments);
  complete();
  EXPECT_FLOAT_EQ(arguments.get_ret<float>(0), 148.0f);
}
}  // namespace taichi::lang::aot_test_utils
