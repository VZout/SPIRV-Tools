// Copyright (c) 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VIK-TODO: BIG ISSUE RIGHT NOW: builder wrapper doesn't work as expected. creating instructions should add them to its parent basic block.
// block::render finalize stack could be removed.

#ifndef SOURCE_STRUCT_RELOOPER_H_
#define SOURCE_STRUCT_RELOOPER_H_

#include <deque>
#include <memory>
#include <vector>

#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.hpp"

// fucky hack to get the last result id
#define BB_INTO_OPERAND(bb) \
  opt::Operand(SPV_OPERAND_TYPE_ID, \
               CreateOperandDataFromU64(bb->end()->result_id()))

#define NULL_OPERAND opt::Operand(SPV_OPERAND_TYPE_NONE, {})

namespace spvtools {
namespace opt {
class Module;
}  // namespace opt
namespace struc {

class Relooper;

class RelooperBuilder : public spvtools::opt::InstructionBuilder {
 public:

  enum class UnaryType {
    AndInt32,
  };
  enum class BinaryType {
    AndInt32,
  };

    std::vector<std::unique_ptr<opt::BasicBlock>>
      trash;  // vector contains trash and code is trash

  // Creates an InstructionBuilder, all new instructions will be inserted before
  // the instruction |insert_before|.
  RelooperBuilder(Relooper* relooper, opt::IRContext* context, opt::Instruction* insert_before,
                  opt::IRContext::Analysis preserved_analyses =
                      opt::IRContext::kAnalysisNone)
      : InstructionBuilder(context, insert_before, preserved_analyses), relooper(relooper) {}

  // Creates an InstructionBuilder, all new instructions will be inserted at the
  // end of the basic block |parent_block|.
  RelooperBuilder(opt::IRContext* context, opt::BasicBlock* parent_block,
                  opt::IRContext::Analysis preserved_analyses =
                      opt::IRContext::kAnalysisNone)
      : InstructionBuilder(context, parent_block, preserved_analyses) {}

  std::unique_ptr<opt::Instruction> NewLabel(uint32_t label_id);

  std::unique_ptr<opt::BasicBlock> MakeNewBlockFromBlock(
      opt::BasicBlock* block);
  opt::BasicBlock* Blockify(opt::BasicBlock* lh, opt::BasicBlock* rh);

  std::uint32_t GetBlockBreakName(std::uint32_t id);

  std::uint32_t MakeType(SpvOp op);
  std::uint32_t MakeLabelType();
  std::uint32_t MakeBoolType();
  std::uint32_t MakeLabelPtrType(std::size_t type_id);
  std::uint32_t MakeConstant(std::size_t type_id, std::size_t value);
  std::unique_ptr<opt::Instruction> MakeLabel();
  std::unique_ptr<opt::Instruction> MakeCheckLabel(std::size_t value);
  std::unique_ptr<opt::Instruction> makeSetLabel(std::size_t value);
  std::unique_ptr<opt::Instruction> makeGetLabel();
  opt::BasicBlock* MakeUnary(UnaryType type, opt::Operand condition); // returning a basic block just to make things easier for me.
  opt::BasicBlock* MakeBinary(
      BinaryType type, opt::Operand lh_cond, opt::Operand rh_cond);  // returning a basic block just
                                                // to make things easier for me.
  // allows creating a conditional branch without a false_branch, this is not allowed in spirv. make sure it always has a false branch with `SetIfFalse`.
  std::unique_ptr<opt::BasicBlock> MakeIf(opt::Operand condition, opt::BasicBlock* true_branch, opt::BasicBlock* false_branch = nullptr);
  void SetIfFalse(
      opt::Instruction* in, opt::BasicBlock* false_branch);
  void SetIfFalse(opt::BasicBlock* in, opt::BasicBlock* false_branch);

  // blockify, but creates a new block instead of appending the first one.
  std::unique_ptr<opt::BasicBlock> MakeSequence(opt::BasicBlock* lh,
                                                opt::BasicBlock* rh);

  // returns the result_id as operand from the lat instruction of the block.
  static opt::Operand OperandFromBasicBlock(opt::BasicBlock* bb);

  std::uint32_t GetUniqueID();

  std::uint32_t label_type_id = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t label_id = std::numeric_limits<std::uint32_t>::max();
  Relooper* relooper;
};

struct Shape;
struct Block;

using Operand = opt::Operand;

// Info about a branching from one block to another
struct Branch {
  enum class FlowType {
    Direct = 0,  // We will directly reach the right location through other
                 // means, no need for continue or break
    Break = 1,
    Continue = 2
  };

  // If not NULL, this shape is the relevant one for purposes of getting to the
  // target block. We break or continue on it
  Shape* Ancestor = nullptr;
  // If Ancestor is not NULL, this says whether to break or continue
  FlowType Type;

  // A branch either has a condition expression if the block ends in ifs, or if
  // the block ends in a switch, then a list of indexes, which becomes the
  // indexes in the table of the switch. If not a switch, the condition can be
  // any expression (or nullptr for the branch taken when no other condition is
  // true) A condition must not have side effects, as the Relooper can reorder
  // or eliminate condition checking. This must not have side effects.
  Operand condition;  // VIK-TODO: renamed this from expression to instruction.
  // This contains the values for which the branch will be taken, or for the
  // default it is simply not present (empty).
  std::vector<std::size_t>
      switch_values;  // VIK-TODO: Using std::size_t instead of wasm::Index.
                      // removed the unique ptr part

  // If provided, code that is run right before the branch is taken. This is
  // useful for phis.
  opt::BasicBlock*
      code;  // VIK-TODO: wasm::Expression original. Interpreted as instruction

  Branch(Operand condition, opt::BasicBlock* code);
  Branch(std::vector<std::size_t> switch_values, opt::BasicBlock* code);

  opt::BasicBlock* Render(RelooperBuilder& builder, Block* target,
                          opt::Function* new_func,
                           bool set_label);
};

// like std::set, except that begin() -> end() iterates in the
// order that elements were added to the set (not in the order
// of operator<(T, T))
template <typename T>
struct InsertOrderedSet {
  std::map<T, typename std::list<T>::iterator> Map;
  std::list<T> List;

  typedef typename std::list<T>::iterator iterator;
  iterator begin() { return List.begin(); }
  iterator end() { return List.end(); }

  void erase(const T& val) {
    auto it = Map.find(val);
    if (it != Map.end()) {
      List.erase(it->second);
      Map.erase(it);
    }
  }

  void erase(iterator position) {
    Map.erase(*position);
    List.erase(position);
  }

  // cheating a bit, not returning the iterator
  void insert(const T& val) {
    auto it = Map.find(val);
    if (it == Map.end()) {
      List.push_back(val);
      Map.insert(std::make_pair(val, --List.end()));
    }
  }

  size_t size() const { return Map.size(); }
  bool empty() const { return Map.empty(); }

  void clear() {
    Map.clear();
    List.clear();
  }

  size_t count(const T& val) const { return Map.count(val); }

  InsertOrderedSet() = default;
  InsertOrderedSet(const InsertOrderedSet& other) { *this = other; }
  InsertOrderedSet& operator=(const InsertOrderedSet& other) {
    clear();
    for (auto i : other.List) {
      insert(i);  // inserting manually creates proper iterators
    }
    return *this;
  }
};

// TODO-VIK: util func
template <class T, class U>
static bool contains(const T& container, const U& contained) {
  return !!container.count(contained);
}

// like std::map, except that begin() -> end() iterates in the
// order that elements were added to the map (not in the order
// of operator<(Key, Key))
template <typename Key, typename T>
struct InsertOrderedMap {
  std::map<Key, typename std::list<std::pair<Key, T>>::iterator> Map;
  std::list<std::pair<Key, T>> List;

  T& operator[](const Key& k) {
    auto it = Map.find(k);
    if (it == Map.end()) {
      List.push_back(std::make_pair(k, T()));
      auto e = --List.end();
      Map.insert(std::make_pair(k, e));
      return e->second;
    }
    return it->second->second;
  }

  typedef typename std::list<std::pair<Key, T>>::iterator iterator;
  iterator begin() { return List.begin(); }
  iterator end() { return List.end(); }

  void erase(const Key& k) {
    auto it = Map.find(k);
    if (it != Map.end()) {
      List.erase(it->second);
      Map.erase(it);
    }
  }

  void erase(iterator position) { erase(position->first); }

  void clear() {
    Map.clear();
    List.clear();
  }

  void swap(InsertOrderedMap<Key, T>& Other) {
    Map.swap(Other.Map);
    List.swap(Other.List);
  }

  size_t size() const { return Map.size(); }
  bool empty() const { return Map.empty(); }
  size_t count(const Key& k) const { return Map.count(k); }

  InsertOrderedMap() = default;
  InsertOrderedMap(InsertOrderedMap& other) {
    abort();  // TODO, watch out for iterators
  }
  InsertOrderedMap& operator=(const InsertOrderedMap& other) {
    abort();  // TODO, watch out for iterators
  }
  bool operator==(const InsertOrderedMap& other) {
    return Map == other.Map && List == other.List;
  }
  bool operator!=(const InsertOrderedMap& other) { return !(*this == other); }
};

using BlockSet = InsertOrderedSet<Block*>;
using BlockBranchMap = InsertOrderedMap<Block*, Branch*>;
using BlockBlockSetMap = InsertOrderedMap<Block*, BlockSet>;
using IdShapeMap = std::map<int, Shape*>;
using BlockBlockMap = std::map<Block*, Block*>;
using BlockList = std::list<Block*>;

// Represents a basic block of code - some instructions that end with a
// control flow modifier (a branch, return or throw).
struct Block {
  Relooper* relooper;
  // Branches become processed after we finish the shape relevant to them. For
  // example, when we recreate a loop, branches to the loop start become
  // continues and are now processed. When we calculate what shape to generate
  // from a set of blocks, we ignore processed branches. Blocks own the Branch
  // objects they use, and destroy them when done.
  BlockBranchMap branches_out;
  BlockSet branches_in;
  BlockBranchMap processed_branches_out;
  BlockSet processed_branches_in;
  Shape* parent = nullptr;  // The shape we are directly inside
  std::uint32_t id = -1;  // A unique identifier, defined when added to relooper
  // The code in this block. This can be arbitrary wasm code, including internal
  // control flow, it should just not branch to the outside
  opt::BasicBlock* code;
  // If nullptr, then this block ends in ifs (or nothing). otherwise, this block
  // ends in a switch, done on this condition
  Operand switch_condition = NULL_OPERAND;
  // If true, we are a multiple entry, so reaching us requires setting the label
  // variable
  bool is_checked_multiple_entry;

  Block(Relooper* relooper, opt::BasicBlock* code, Operand switch_condition = NULL_OPERAND);
  ~Block();

  // Add a branch: if the condition holds we branch (or if null, we branch if
  // all others failed) Note that there can be only one branch from A to B (if
  // you need multiple conditions for the branch, create a more interesting
  // expression in the Condition). If a Block has no outgoing branches, the
  // contents in Code must contain a terminating instruction, as the relooper
  // doesn't know whether you want control flow to stop with an `unreachable` or
  // a `return` or something else (if you forget to do this, control flow may
  // continue into the block that happens to be emitted right after it).
  // Internally, adding a branch only adds the outgoing branch. The matching
  // incoming branch on the target is added by the Relooper itself as it works.
  void AddBranchTo(Block* target, Operand condition,
                   opt::BasicBlock* code = nullptr);

  // Add a switch branch: if the switch condition is one of these values, we
  // branch (or if the list is empty, we are the default) Note that there can be
  // only one branch from A to B (if you need multiple values for the branch,
  // that's what the array and default are for).
  void AddSwitchBranchTo(Block* target, std::vector<std::size_t>&& values,
                         opt::BasicBlock* code = nullptr);

  opt::BasicBlock* Render(RelooperBuilder& builder,
                                          opt::Function* new_func,
                                          bool in_loop);
};

struct SimpleShape;
struct MultipleShape;
struct LoopShape;

struct Shape {
  enum class Type { Simple, Multiple, Loop };

  Shape(Type type) : type(type) {}
  virtual ~Shape() = default;

  virtual opt::BasicBlock* Render(RelooperBuilder& builder,
                                  opt::Function* new_func, bool in_loop) = 0;

  static SimpleShape* IsSimple(Shape* it) {
    return it && it->type == Type::Simple ? (SimpleShape*)it : NULL;
  }
  static MultipleShape* IsMultiple(Shape* it) {
    return it && it->type == Type::Multiple ? (MultipleShape*)it : NULL;
  }
  static LoopShape* IsLoop(Shape* it) {
    return it && it->type == Type::Loop ? (LoopShape*)it : NULL;
  }

  // A unique identifier. Used to identify loops, labels are Lx where x is the
  // Id. Defined when added to relooper
  std::size_t id = -1;
  // The shape that will appear in the code right after this one
  Shape* next = nullptr;
  // The shape that control flow gets to naturally (if there is Next, then this
  // is Next)
  Shape* natural = nullptr;
  Type type;
};

struct SimpleShape : public Shape {
  SimpleShape() : Shape(Type::Simple) {}
  opt::BasicBlock* Render(RelooperBuilder& builder, opt::Function* new_func,
                          bool in_loop) override;

  Block* inner = nullptr;
};

struct MultipleShape : public Shape {
  MultipleShape() : Shape(Type::Multiple) {}
  opt::BasicBlock* Render(RelooperBuilder& builder, opt::Function* new_func,
                          bool in_loop) override;

  IdShapeMap inner_map;  // entry block ID -> shape
};

struct LoopShape : public Shape {
  LoopShape() : Shape(Type::Loop) {}
  opt::BasicBlock* Render(RelooperBuilder& builder, opt::Function* new_func,
                          bool in_loop) override;

  Shape* inner = nullptr;
  BlockSet entries;  // we must visit at least one of these
};

class Relooper {
 public:
  Relooper(opt::IRContext* context);
  ~Relooper();

  void AddUsedID(std::uint32_t id);
  std::vector<std::uint32_t> used_ids;
  std::uint32_t GetUniqueID();

  // Disables copy/move constructor/assignment operations.
  Relooper(const Relooper&) = delete;
  Relooper(Relooper&&) = delete;
  Relooper& operator=(const Relooper&) = delete;
  Relooper& operator=(Relooper&&) = delete;

  void Calculate(Block* entry);
  std::unique_ptr<opt::Function> Render(opt::IRContext* new_context,
                                        opt::Function& old_function);

  Branch* AddBranch(Operand condition, opt::BasicBlock* code);

  Block* NewBlock();
    Block* AddBlock(opt::BasicBlock* code, opt::BasicBlock* switch_condition);

  template <typename T>
  T* AddShape() {
    auto shape = std::make_unique<T>();
    shape->id = shape_id_counter++;
    auto* shapePtr = shape.get();
    shapes.push_back(std::move(shape));
    return shapePtr;
  }

  opt::IRContext* GetContext() { return context;
  }

  void ForEachBlock(std::function<void(Block*)> lambda);

 private:
  opt::IRContext* context;
  std::deque<std::unique_ptr<Block>> blocks;
  std::deque<std::unique_ptr<Shape>> shapes;
  std::deque<std::unique_ptr<Branch>> branches;
  Shape* root;
  bool min_size;
  std::uint32_t block_id_counter;
  std::size_t shape_id_counter;
};

}  // namespace struc
}  // namespace spvtools

#endif  // SOURCE_STRUCT_RELOOPER_H_