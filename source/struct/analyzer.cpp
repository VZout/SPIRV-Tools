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

#include "analyzer.h"

namespace spvtools {
namespace struc {

void Analyzer::GetBlocksOut(Block* source, BlockSet& entries,
                            BlockSet* limit_to) {
  for (auto& iter : source->branches_out) {
    if (!limit_to || contains(*limit_to, iter.first)) {
      entries.insert(iter.first);
    }
  }
}

void Analyzer::Solipsize(Block* target, Branch::FlowType type, Shape* ancestor,
                         BlockSet& from) {
  printf("Solipsizing branches into %d\n", target->id);
  // DebugDump(From, "  relevant to solipsize: ");
  for (auto iter = target->branches_in.begin();
       iter != target->branches_in.end();) {
    Block* Prior = *iter;
    if (!contains(from, Prior)) {
      iter++;
      continue;
    }
    Branch* PriorOut = Prior->branches_out[target];
    PriorOut->Ancestor = ancestor;
    PriorOut->Type = type;
    iter++;  // carefully increment iter before erasing
    target->branches_in.erase(Prior);
    target->processed_branches_in.insert(Prior);
    Prior->branches_out.erase(target);
    Prior->processed_branches_out[target] = PriorOut;
    printf("  eliminated branch from %d\n", Prior->id);
  }
}

Shape* Analyzer::MakeSimple(BlockSet& blocks, Block* inner,
                            BlockSet& next_entries) {
  printf("creating simple block with block #%d\n", inner->id);
  SimpleShape* Simple = parent->AddShape<SimpleShape>();
  Simple->inner = inner;

  
  if (inner->id == 9) {
    inner->code->ForEachInst([&](auto inst) {
      std::cout << "whohay" << inst->opcode() << std::endl;
    });
  }


  inner->parent = Simple;
  if (blocks.size() > 1) {
    blocks.erase(inner);
    GetBlocksOut(inner, next_entries, &blocks);
    BlockSet JustInner;
    JustInner.insert(inner);
    for (auto* Next : next_entries) {
      Solipsize(Next, Branch::FlowType::Break, Simple, JustInner);
    }
  }
  return Simple;
}

// loop is going to be problematic. ill likely have to inject extra blocks into this myself.
Shape* Analyzer::MakeLoop(BlockSet& blocks, BlockSet& entries,
                          BlockSet& next_entries) {
  // Find the inner blocks in this loop. Proceed backwards from the entries
  // until you reach a seen block, collecting as you go.
  BlockSet InnerBlocks;
  BlockSet Queue = entries;
  while (Queue.size() > 0) {
    Block* Curr = *(Queue.begin());
    Queue.erase(Queue.begin());
    if (!contains(InnerBlocks, Curr)) {
      // This element is new, mark it as inner and remove from outer
      InnerBlocks.insert(Curr);
      blocks.erase(Curr);
      // Add the elements prior to it
      for (auto* Prev : Curr->branches_in) {
        Queue.insert(Prev);
      }
    }
  }
  assert(InnerBlocks.size() > 0);

  for (auto* Curr : InnerBlocks) {
    for (auto& iter : Curr->branches_out) {
      Block* Possible = iter.first;
      if (!contains(InnerBlocks, Possible)) {
        next_entries.insert(Possible);
      }
    }
  }
  printf("creating loop block:\n");
  /*DebugDump(InnerBlocks, "  inner blocks:");
  DebugDump(Entries, "  inner entries:");
  DebugDump(Blocks, "  outer blocks:");
  DebugDump(NextEntries, "  outer entries:");*/

  LoopShape* Loop = parent->AddShape<LoopShape>();

  // Solipsize the loop, replacing with break/continue and marking branches
  // as Processed (will not affect later calculations) A. Branches to the
  // loop entries become a continue to this shape
  for (auto* Entry : entries) {
    Solipsize(Entry, Branch::FlowType::Continue, Loop, InnerBlocks);
  }
  // B. Branches to outside the loop (a next entry) become breaks on this
  // shape
  for (auto* Next : next_entries) {
    Solipsize(Next, Branch::FlowType::Break, Loop, InnerBlocks);
  }

  Shape* Inner = Process(InnerBlocks, entries);
  Loop->inner = Inner;
  Loop->entries = entries;
  return Loop;
}

Shape* Analyzer::MakeMultiple(BlockSet& blocks, BlockSet& entries,
                              BlockBlockSetMap& IndependentGroups,
                              BlockSet& next_entries,
                              bool is_checked_multiple) {
  printf("creating multiple block with %d inner groups\n",
             (int)IndependentGroups.size());
  MultipleShape* Multiple = parent->AddShape<MultipleShape>();
  BlockSet CurrEntries;
  for (auto& iter : IndependentGroups) {
    Block* CurrEntry = iter.first;
    BlockSet& CurrBlocks = iter.second;
    // PrintDebug("  multiple group with entry %d:\n", CurrEntry->Id);
    // DebugDump(CurrBlocks, "    ");
    // Create inner block
    CurrEntries.clear();
    CurrEntries.insert(CurrEntry);
    for (auto* CurrInner : CurrBlocks) {
      // Remove the block from the remaining blocks
      blocks.erase(CurrInner);
      // Find new next entries and fix branches to them
      for (auto iter = CurrInner->branches_out.begin();
           iter != CurrInner->branches_out.end();) {
        Block* CurrTarget = iter->first;
        auto Next = iter;
        Next++;
        if (!contains(CurrBlocks, CurrTarget)) {
          next_entries.insert(CurrTarget);
          Solipsize(CurrTarget, Branch::FlowType::Break, Multiple, CurrBlocks);
        }
        iter = Next;  // increment carefully because Solipsize can remove us
      }
    }
    Multiple->inner_map[CurrEntry->id] = Process(CurrBlocks, CurrEntries);
    if (is_checked_multiple) {
      CurrEntry->is_checked_multiple_entry = true;
    }
  }
  // DebugDump(Blocks, "  remaining blocks after multiple:");
  // Add entries not handled as next entries, they are deferred
  for (auto* Entry : entries) {
    if (!contains(IndependentGroups, Entry)) {
      next_entries.insert(Entry);
    }
  }
  return Multiple;
}

void Analyzer::FindIndependentGroups(BlockSet& entries,
                                     BlockBlockSetMap& independent_groups,
                                     BlockSet* Ignore) {
  // We flow out from each of the entries, simultaneously.
  // When we reach a new block, we add it as belonging to the one we got to
  // it from. If we reach a new block that is already marked as belonging to
  // someone, it is reachable by two entries and is not valid for any of
  // them. Remove it and all it can reach that have been visited.

  // Being in the queue means we just added this item, and we need to add
  // its children
  BlockList Queue;
  BlockBlockMap Ownership;
  for (auto* Entry : entries) {
    Ownership[Entry] = Entry;
    independent_groups[Entry].insert(Entry);
    Queue.push_back(Entry);
  }
  while (Queue.size() > 0) {
    Block* Curr = Queue.front();
    Queue.pop_front();
    // Curr must be in the ownership map if we are in the queue
    Block* Owner = Ownership[Curr];
    if (!Owner) {
      // we have been invalidated meanwhile after being reached from two
      // entries
      continue;
    }
    // Add all children
    for (auto& iter : Curr->branches_out) {
      Block* New = iter.first;
      auto Known = Ownership.find(New);
      if (Known == Ownership.end()) {
        // New node. Add it, and put it in the queue
        Ownership[New] = Owner;
        independent_groups[Owner].insert(New);
        Queue.push_back(New);
        continue;
      }
      Block* NewOwner = Known->second;
      if (!NewOwner) {
        continue;  // We reached an invalidated node
      }
      if (NewOwner != Owner) {
        // Invalidate this and all reachable that we have seen - we reached
        // this from two locations
        InvalidateWithChildren(New, independent_groups, Ownership);
      }
      // otherwise, we have the same owner, so do nothing
    }
  }

  // Having processed all the interesting blocks, we remain with just one
  // potential issue: If a->b, and a was invalidated, but then b was later
  // reached by someone else, we must invalidate b. To check for this, we go
  // over all elements in the independent groups, if an element has a parent
  // which does *not* have the same owner, we must remove it and all its
  // children.

  for (auto* Entry : entries) {
    BlockSet& CurrGroup = independent_groups[Entry];
    BlockList ToInvalidate;
    for (auto* Child : CurrGroup) {
      for (auto* Parent : Child->branches_in) {
        if (Ignore && contains(*Ignore, Parent)) {
          continue;
        }
        if (Ownership[Parent] != Ownership[Child]) {
          ToInvalidate.push_back(Child);
        }
      }
    }
    while (ToInvalidate.size() > 0) {
      Block* Invalidatee = ToInvalidate.front();
      ToInvalidate.pop_front();
      InvalidateWithChildren(Invalidatee, independent_groups, Ownership);
    }
  }

  // Remove empty groups
  for (auto* Entry : entries) {
    if (independent_groups[Entry].size() == 0) {
      independent_groups.erase(Entry);
    }
  }

#ifdef RELOOPER_DEBUG
  PrintDebug("Investigated independent groups:\n");
  for (auto& iter : IndependentGroups) {
    DebugDump(iter.second, " group: ");
  }
#endif
}

void Analyzer::InvalidateWithChildren(Block* New,
                                      BlockBlockSetMap& independent_groups,
                                      BlockBlockMap& ownership) {
  // Being in the list means you need to be invalidated
  BlockList ToInvalidate;
  ToInvalidate.push_back(New);
  while (ToInvalidate.size() > 0) {
    Block* Invalidatee = ToInvalidate.front();
    ToInvalidate.pop_front();
    Block* Owner = ownership[Invalidatee];
    // Owner may have been invalidated, do not add to IndependentGroups!
    if (contains(independent_groups, Owner)) {
      independent_groups[Owner].erase(Invalidatee);
    }
    // may have been seen before and invalidated already
    if (ownership[Invalidatee]) {
      ownership[Invalidatee] = nullptr;
      for (auto& iter : Invalidatee->branches_out) {
        Block* Target = iter.first;
        auto Known = ownership.find(Target);
        if (Known != ownership.end()) {
          Block* TargetOwner = Known->second;
          if (TargetOwner) {
            ToInvalidate.push_back(Target);
          }
        }
      }
    }
  }
}

Shape* Analyzer::Process(BlockSet& blocks, BlockSet& initialEntries) {
  // PrintDebug("Process() called\n", 0);
  BlockSet* Entries = &initialEntries;
  BlockSet TempEntries[2];
  int CurrTempIndex = 0;
  BlockSet* NextEntries;
  Shape* Ret = nullptr;
  Shape* Prev = nullptr;
#define Make(call)                              \
  Shape* Temp = call;                           \
  if (Prev) Prev->next = Temp;                  \
  if (!Ret) Ret = Temp;                         \
  if (!NextEntries->size()) {                   \
    /*PrintDebug("Process() returning\n", 0);*/ \
    return Ret;                                 \
  }                                             \
  Prev = Temp;                                  \
  Entries = NextEntries;                        \
  continue;
  while (1) {
    // PrintDebug("Process() running\n", 0);
    // DebugDump(Blocks, "  blocks : ");
    // DebugDump(*Entries, "  entries: ");

    CurrTempIndex = 1 - CurrTempIndex;
    NextEntries = &TempEntries[CurrTempIndex];
    NextEntries->clear();

    if (Entries->size() == 0) {
      return Ret;
    }
    if (Entries->size() == 1) {
      Block* Curr = *(Entries->begin());
      if (Curr->branches_in.size() == 0) {
        // One entry, no looping ==> Simple
        Make(MakeSimple(blocks, Curr, *NextEntries));
      }
      // One entry, looping ==> Loop
      Make(MakeLoop(blocks, *Entries, *NextEntries));
    }

    // More than one entry, try to eliminate through a Multiple groups of
    // independent blocks from an entry/ies. It is important to remove
    // through multiples as opposed to looping since the former is more
    // performant.
    BlockBlockSetMap IndependentGroups;
    FindIndependentGroups(*Entries, IndependentGroups);

    // PrintDebug("Independent groups: %d\n", IndependentGroups.size());

    if (IndependentGroups.size() > 0) {
      // We can handle a group in a multiple if its entry cannot be reached
      // by another group. Note that it might be reachable by itself - a
      // loop. But that is fine, we will create a loop inside the multiple
      // block, which is both the performant order to do it, and preserves
      // the property that a loop will always reach an entry.
      for (auto iter = IndependentGroups.begin();
           iter != IndependentGroups.end();) {
        Block* Entry = iter->first;
        BlockSet& Group = iter->second;
        auto curr = iter++;  // iterate carefully, we may delete
        for (auto iterBranch = Entry->branches_in.begin();
             iterBranch != Entry->branches_in.end(); iterBranch++) {
          Block* Origin = *iterBranch;
          if (!contains(Group, Origin)) {
            // Reached from outside the group, so we cannot handle this
            /*PrintDebug(
                "Cannot handle group with entry %d because of "
                "incoming branch from %d\n",
                Entry->Id, Origin->Id);*/
            IndependentGroups.erase(curr);
            break;
          }
        }
      }

      // As an optimization, if we have 2 independent groups, and one is a
      // small dead end, we can handle only that dead end. The other then
      // becomes a Next - without nesting in the code and recursion in the
      // analysis.
      // TODO: if the larger is the only dead end, handle that too
      // TODO: handle >2 groups
      // TODO: handle not just dead ends, but also that do not branch to the
      //       NextEntries. However, must be careful
      //       there since we create a Next, and that Next can prevent
      //       eliminating a break (since we no longer naturally reach the
      //       same place), which may necessitate a one-time loop, which
      //       makes the unnesting pointless.
      if (IndependentGroups.size() == 2) {
        // Find the smaller one
        auto iter = IndependentGroups.begin();
        Block* SmallEntry = iter->first;
        std::size_t SmallSize = iter->second.size();
        iter++;
        Block* LargeEntry = iter->first;
        std::size_t LargeSize = iter->second.size();
        // ignore the case where they are identical - keep things
        // symmetrical there
        if (SmallSize != LargeSize) {
          if (SmallSize > LargeSize) {
            Block* Temp = SmallEntry;
            SmallEntry = LargeEntry;
            // Note: we did not flip the Sizes too, they are now invalid.
            // TODO: use the smaller size as a limit?
            LargeEntry = Temp;
          }
          // Check if dead end
          bool DeadEnd = true;
          BlockSet& SmallGroup = IndependentGroups[SmallEntry];
          for (auto* Curr : SmallGroup) {
            for (auto& iter : Curr->branches_out) {
              Block* Target = iter.first;
              if (!contains(SmallGroup, Target)) {
                DeadEnd = false;
                break;
              }
            }
            if (!DeadEnd) {
              break;
            }
          }
          if (DeadEnd) {
            /*PrintDebug(
                "Removing nesting by not handling large group "
                "because small group is dead end\n",
                0);
            IndependentGroups.erase(LargeEntry);*/
          }
        }
      }

      /*PrintDebug("Handleable independent groups: %d\n",
                 IndependentGroups.size());*/

      if (IndependentGroups.size() > 0) {
        // Some groups removable ==> Multiple
        // This is a checked multiple if it has an entry that is an entry to
        // this Process call, that is, if we can reach it from outside this
        // set of blocks, then we must check the label variable to do so.
        // Otherwise, if it is just internal blocks, those can always be
        // jumped to forward, without using the label variable
        bool Checked = false;
        for (auto* Entry : *Entries) {
          if (initialEntries.count(Entry)) {
            Checked = true;
            break;
          }
        }
        Make(MakeMultiple(blocks, *Entries, IndependentGroups, *NextEntries,
                          Checked));
      }
    }
    // No independent groups, must be loopable ==> Loop
    Make(MakeLoop(blocks, *Entries, *NextEntries));
  }
}

}  // namespace struc
}  // namespace spvtools