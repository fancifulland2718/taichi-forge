#pragma once

// Specialized Attributes and functions
struct BitmaskedMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(BitmaskedMeta, _);

i32 Bitmasked_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void Bitmasked_activate(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Bitmasked_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = (u32 *)(node + data_section_size);
  u32 bit = 1UL << (i % 32);
  u32 prev = atomic_or_u32(&mask_begin[i / 32], bit);
  if ((prev & bit) == 0) {
    mark_element_lists_dirty_if_reuse(smeta);
  }
}

void Bitmasked_deactivate(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Bitmasked_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = (u32 *)(node + data_section_size);
  u32 bit = 1UL << (i % 32);
  u32 prev = atomic_and_u32(&mask_begin[i / 32], ~bit);
  if (prev & bit) {
    mark_element_lists_dirty_if_reuse(smeta);
  }
}

// G11-A (2026-05): Bitmasked_deactivate 的"清 data slot"变体。仅在
// CompileConfig::bitmasked_clear_data_on_deactivate=true 时被 codegen
// 选用（codegen_llvm.cpp 路由）。__atomic_fetch_and 返回旧 mask word，
// 只有"真翻 1→0"的那个唯一线程（prev & bit 非 0）才执行 memset，
// 避免重复清零。读者必须先看到 mask=0（=> is_active=false）才会避
// 让出该 cell；mask 翻 0 之前的并发 atomic 写入仍按原语义生效；
// 翻 0 之后到 memset 完成之间，没有合法读者会访问该 cell（此时
// is_active=false，runtime 路径直接跳过），故清零序列无 race。
void Bitmasked_deactivate_and_clear(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Bitmasked_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = (u32 *)(node + data_section_size);
  u32 bit = 1UL << (i % 32);
  u32 prev = atomic_and_u32(&mask_begin[i / 32], ~bit);
  if (prev & bit) {
    mark_element_lists_dirty_if_reuse(smeta);
    std::memset(node + (std::size_t)element_size * (std::size_t)i, 0,
                element_size);
  }
}

u1 Bitmasked_is_active(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Dense_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = node + data_section_size;
  return bool((mask_begin[i / 8] >> (i % 8)) & 1);
}

Ptr Bitmasked_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}
