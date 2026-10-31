#include "gtest/gtest.h"

#include "taichi/ir/type_factory.h"

namespace taichi::lang {

TEST(Type, StructOffsetsPreserveImplicitAndDeclaredLayouts) {
  auto &factory = TypeFactory::get_instance();
  const auto *vector = factory.get_tensor_type({3}, PrimitiveType::f32);
  const auto *implicit = factory.get_struct_type(
      {{PrimitiveType::i8, "tag"}, {PrimitiveType::f64, "value"},
       {vector, "vector"}})->as<StructType>();
  EXPECT_EQ(implicit->get_element_offset({1}), 8);
  EXPECT_EQ(implicit->get_element_offset({2, 2}), 24);

  // A later explicit offset determines the policy for the entire structure,
  // including earlier zero offsets. Do not stop at the queried member.
  const auto *explicit_layout = factory.get_struct_type(
      {{PrimitiveType::i32, "first"}, {PrimitiveType::i32, "second"},
       {PrimitiveType::i32, "last", 32}})->as<StructType>();
  EXPECT_EQ(explicit_layout->get_element_offset({1}), 0);
  EXPECT_EQ(explicit_layout->get_element_offset({2}), 32);

  const auto *pointer = factory.get_struct_type(
      {{factory.get_pointer_type(PrimitiveType::f32), "data"}});
  const auto *nested_pointer = factory.get_struct_type(
      {{PrimitiveType::i32, "first"}, {PrimitiveType::i32, "second"},
       {pointer, "nested"}})->as<StructType>();
  EXPECT_EQ(nested_pointer->get_element_offset({1}), 0);
  EXPECT_EQ(nested_pointer->get_element_offset({2, 0}), 0);

  std::vector<AbstractDictionaryMember> members;
  for (std::size_t i = 0; i < 512; ++i)
    members.push_back({pointer, "array" + std::to_string(i), i * 16});
  const auto *wide = factory.get_struct_type(members)->as<StructType>();
  for (int i : {0, 1, 63, 511})
    EXPECT_EQ(wide->get_element_offset({i, 0}), i * 16);
}

TEST(Type, TypeToString) {
  auto f16 = TypeFactory::get_instance().get_primitive_real_type(16);
  EXPECT_EQ(f16->to_string(), "f16");

  auto f32 = TypeFactory::get_instance().get_primitive_real_type(32);
  auto i32 = TypeFactory::get_instance().get_primitive_int_type(32, true);
  auto u32 = TypeFactory::get_instance().get_primitive_int_type(32, false);
  auto u16 = TypeFactory::get_instance().get_primitive_int_type(16, false);
  auto qi5 = TypeFactory::get_instance().get_quant_int_type(5, true, i32);
  auto qu7 = TypeFactory::get_instance().get_quant_int_type(7, false, i32);
  auto qfl = TypeFactory::get_instance().get_quant_float_type(qi5, qu7, f32);

  auto bs1 = TypeFactory::get_instance().get_bit_struct_type(
      /*physical_type=*/u16, /*member_types=*/{qi5, qu7},
      /*member_bit_offsets=*/{0, 5}, /*member_exponents=*/{-1, -1},
      /*member_exponent_users=*/{{}, {}});
  EXPECT_EQ(bs1->to_string(), "bs(0: qi5@0, 1: qu7@5)");

  auto bs2 = TypeFactory::get_instance().get_bit_struct_type(
      /*physical_type=*/u32, /*member_types=*/{qu7, qfl, qu7, qfl},
      /*member_bit_offsets=*/{0, 7, 12, 19},
      /*member_exponents=*/{-1, 0, -1, 2},
      /*member_exponent_users=*/{{1}, {}, {3}, {}});
  EXPECT_EQ(bs2->to_string(),
            "bs(0: qu7@0, 1: qfl(d=qi5 e=qu7 c=f32)@7 exp=0, 2: qu7@12, 3: "
            "qfl(d=qi5 e=qu7 c=f32)@19 exp=2)");

  auto bs3 = TypeFactory::get_instance().get_bit_struct_type(
      /*physical_type=*/u32, /*member_types=*/{qu7, qfl, qfl},
      /*member_bit_offsets=*/{0, 7, 12}, /*member_exponents=*/{-1, 0, 0},
      /*member_exponent_users=*/{{1, 2}, {}, {}});
  EXPECT_EQ(bs3->to_string(),
            "bs(0: qu7@0, 1: qfl(d=qi5 e=qu7 c=f32)@7 shared_exp=0, 2: "
            "qfl(d=qi5 e=qu7 c=f32)@12 shared_exp=0)");

  auto qi1 = TypeFactory::get_instance().get_quant_int_type(1, true, i32);
  auto qa = TypeFactory::get_instance().get_quant_array_type(i32, qi1, 32);
  EXPECT_EQ(qa->to_string(), "qa(qi1x32)");
}

}  // namespace taichi::lang
