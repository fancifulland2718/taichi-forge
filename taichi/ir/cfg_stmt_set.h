#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>

namespace taichi::lang {

class Stmt;

// Dense dataflow facts share one statement numbering per analysis. A bit per
// fact avoids a separately allocated hash node for every (CFG node, fact).
// Finish numbering before reset(); set operations require the same universe.
class CFGStmtSet {
 public:
  struct Universe {
    std::vector<Stmt *> statements;
    std::unordered_map<Stmt *, std::size_t> indices;

    void insert(Stmt *stmt) {
      if (indices.emplace(stmt, statements.size()).second)
        statements.push_back(stmt);
    }
  };

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Stmt *;
    using difference_type = std::ptrdiff_t;
    using pointer = Stmt *const *;
    using reference = Stmt *const &;

    const_iterator() = default;
    const_iterator(const CFGStmtSet *set, std::size_t index)
        : set_(set), index_(index) {
      advance();
    }
    reference operator*() const {
      return set_->universe_->statements[index_];
    }
    const_iterator &operator++() {
      ++index_;
      advance();
      return *this;
    }
    const_iterator operator++(int) {
      auto old = *this;
      ++*this;
      return old;
    }
    bool operator==(const const_iterator &other) const {
      return set_ == other.set_ && index_ == other.index_;
    }
    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

   private:
    void advance() {
      const auto end = set_->universe_ ? set_->universe_->statements.size() : 0;
      while (index_ < end) {
        auto word = set_->words_[index_ / 64] & (~uint64_t(0) << (index_ % 64));
        if (!word) {
          index_ = (index_ / 64 + 1) * 64;
          continue;
        }
        // Binary search the first set bit without backend/compiler intrinsics.
        unsigned bit = 0;
        for (unsigned shift : {32u, 16u, 8u, 4u, 2u, 1u}) {
          if (!(word & ((uint64_t(1) << shift) - 1))) {
            word >>= shift;
            bit += shift;
          }
        }
        index_ = (index_ / 64) * 64 + bit;
        return;
      }
      index_ = end;
    }
    const CFGStmtSet *set_{nullptr};
    std::size_t index_{0};
  };

  void reset(std::shared_ptr<const Universe> universe) {
    universe_ = std::move(universe);
    words_.assign((universe_->statements.size() + 63) / 64, 0);
  }
  void clear() {
    std::fill(words_.begin(), words_.end(), 0);
  }
  void insert(Stmt *stmt) {
    insert_index(universe_->indices.at(stmt));
  }
  void insert_index(std::size_t index) {
    words_[index / 64] |= uint64_t(1) << (index % 64);
  }
  bool contains(Stmt *stmt) const {
    if (!universe_)
      return false;
    auto it = universe_->indices.find(stmt);
    return it != universe_->indices.end() && contains_index(it->second);
  }
  bool contains_index(std::size_t index) const {
    return (words_[index / 64] & (uint64_t(1) << (index % 64))) != 0;
  }
  void unite(const CFGStmtSet &other) {
    for (std::size_t i = 0; i < words_.size(); ++i)
      words_[i] |= other.words_[i];
  }
  // Assign gen U (input - kill), reporting whether the fixed-point changed.
  bool transfer(const CFGStmtSet &input,
                const CFGStmtSet &gen,
                const CFGStmtSet &kill) {
    bool changed = false;
    for (std::size_t i = 0; i < words_.size(); ++i) {
      auto value = gen.words_[i] | (input.words_[i] & ~kill.words_[i]);
      changed |= words_[i] != value;
      words_[i] = value;
    }
    return changed;
  }
  bool empty() const {
    return std::all_of(words_.begin(), words_.end(),
                       [](uint64_t word) { return word == 0; });
  }
  std::size_t size() const {
    std::size_t result = 0;
    for (auto word : words_) {
      while (word) {
        word &= word - 1;
        ++result;
      }
    }
    return result;
  }
  std::size_t storage_bytes() const {
    return words_.size() * sizeof(uint64_t);
  }
  const_iterator begin() const {
    return {this, 0};
  }
  const_iterator end() const {
    return {this, universe_ ? universe_->statements.size() : 0};
  }

 private:
  std::shared_ptr<const Universe> universe_;
  std::vector<uint64_t> words_;
};

}  // namespace taichi::lang
