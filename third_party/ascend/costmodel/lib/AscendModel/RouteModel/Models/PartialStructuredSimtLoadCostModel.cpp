#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadCostModel.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::ascend;
namespace {
// Evaluate only constant/range mask graphs, never loaded or runtime values.
struct Tensor {
  SmallVector<int64_t> shape, values;
};
class StaticMask {
  llvm::DenseMap<Value, std::optional<Tensor>> memo;

public:
  std::optional<Tensor> evaluate(Value value) {
    if (auto it = memo.find(value); it != memo.end())
      return it->second;
    auto result = compute(value);
    memo[value] = result;
    return result;
  }

private:
  std::optional<Tensor> compute(Value value) {
    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    Tensor result;
    int64_t count = 1;
    Type scalarType = value.getType();
    if (auto t = dyn_cast<RankedTensorType>(value.getType())) {
      if (!t.hasStaticShape() || t.getNumElements() > 2048)
        return std::nullopt;
      result.shape.assign(t.getShape().begin(), t.getShape().end());
      count = t.getNumElements();
      scalarType = t.getElementType();
    }
    if (auto integer = dyn_cast<IntegerType>(scalarType)) {
      if (integer.getWidth() > 64)
        return std::nullopt;
    } else if (!isa<IndexType>(scalarType))
      return std::nullopt;
    auto name = op->getName().getStringRef();
    if (name == "arith.constant") {
      if (auto a = op->getAttrOfType<IntegerAttr>("value"))
        result.values.assign(count, a.getValue().isOne() ? 1 : a.getInt());
      else if (auto a = op->getAttrOfType<DenseIntElementsAttr>("value"))
        for (const auto &v : a.getValues<APInt>())
          result.values.push_back(v.isOne() ? 1 : v.getSExtValue());
      else
        return std::nullopt;
      return result;
    }
    if (name == "tt.make_range") {
      auto start = op->getAttrOfType<IntegerAttr>("start");
      if (!start)
        return std::nullopt;
      for (int64_t i = 0; i < count; ++i)
        result.values.push_back(start.getInt() + i);
      return result;
    }
    if (op->getNumOperands() == 1 &&
        llvm::is_contained({StringRef("tt.splat"), StringRef("tt.broadcast"),
                            StringRef("tt.expand_dims"),
                            StringRef("tt.reshape"), StringRef("arith.extsi"),
                            StringRef("arith.extui"),
                            StringRef("arith.index_cast")},
                           name)) {
      auto src = evaluate(op->getOperand(0));
      if (!src)
        return std::nullopt;
      // Values are represented as signed int64_t, except i1 predicates use
      // 0/1. Do not treat zero extension of negative values, or sign extension
      // of true i1, as an identity: that can prove a false rectangle.
      if (name == "arith.extui" &&
          llvm::any_of(src->values, [](int64_t v) { return v < 0; }))
        return std::nullopt;
      Type sourceType = op->getOperand(0).getType();
      if (auto tensor = dyn_cast<RankedTensorType>(sourceType))
        sourceType = tensor.getElementType();
      if (name == "arith.extsi" && sourceType.isInteger(1) &&
          llvm::any_of(src->values, [](int64_t v) { return v != 0; }))
        return std::nullopt;
      if (name != "tt.broadcast" && name != "tt.splat") {
        if (int64_t(src->values.size()) != count)
          return std::nullopt;
        result.values = src->values;
      } else {
        for (int64_t t = 0; t < count; ++t) {
          int64_t flat = t, from = 0, stride = 1;
          for (size_t i = result.shape.size(); i-- > 0;) {
            int64_t j = flat % result.shape[i];
            flat /= result.shape[i];
            if (i < src->shape.size()) {
              if (src->shape[i] != 1 && src->shape[i] != result.shape[i])
                return std::nullopt;
              from += (src->shape[i] == 1 ? 0 : j) * stride;
              stride *= src->shape[i];
            }
          }
          result.values.push_back(src->values[from]);
        }
      }
      return result;
    }
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto a = evaluate(op->getOperand(0)), b = evaluate(op->getOperand(1));
    if (!a || !b || a->values.size() != size_t(count) ||
        b->values.size() != size_t(count))
      return std::nullopt;
    for (int64_t i = 0; i < count; ++i) {
      const int64_t x = a->values[i], y = b->values[i];
      int64_t z = 0;
      if (name == "arith.addi") {
        if (__builtin_add_overflow(x, y, &z))
          return std::nullopt;
      } else if (name == "arith.subi") {
        if (__builtin_sub_overflow(x, y, &z))
          return std::nullopt;
      } else if (name == "arith.muli") {
        if (__builtin_mul_overflow(x, y, &z))
          return std::nullopt;
      } else if (name == "arith.andi")
        z = x & y;
      else if (name == "arith.ori")
        z = x | y;
      else if (name == "arith.divsi" || name == "arith.remsi" ||
               name == "arith.divui" || name == "arith.remui") {
        if (y <= 0 || x < 0)
          return std::nullopt;
        z = name.contains("div") ? x / y : x % y;
      } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq:
          z = x == y;
          break;
        case arith::CmpIPredicate::ne:
          z = x != y;
          break;
        case arith::CmpIPredicate::slt:
          z = x < y;
          break;
        case arith::CmpIPredicate::sle:
          z = x <= y;
          break;
        case arith::CmpIPredicate::sgt:
          z = x > y;
          break;
        case arith::CmpIPredicate::sge:
          z = x >= y;
          break;
        case arith::CmpIPredicate::ult:
          z = uint64_t(x) < uint64_t(y);
          break;
        case arith::CmpIPredicate::ule:
          z = uint64_t(x) <= uint64_t(y);
          break;
        case arith::CmpIPredicate::ugt:
          z = uint64_t(x) > uint64_t(y);
          break;
        case arith::CmpIPredicate::uge:
          z = uint64_t(x) >= uint64_t(y);
          break;
        }
      } else
        return std::nullopt;
      Type scalarType = value.getType();
      if (auto tensor = dyn_cast<RankedTensorType>(scalarType))
        scalarType = tensor.getElementType();
      if (auto integer = dyn_cast<IntegerType>(scalarType)) {
        const unsigned bits = integer.getWidth();
        if (bits > 64 || (bits > 1 && bits < 64 &&
                          (z < -(int64_t(1) << (bits - 1)) ||
                           z > (int64_t(1) << (bits - 1)) - 1)))
          return std::nullopt;
      }
      result.values.push_back(z);
    }
    return result;
  }
};

// Extract the explicit additive program_id term. A loaded index remains an
// opaque residual even if its table address itself contains program_id.
struct AffineProgram {
  int64_t constant = 0, coefficient = 0;
  bool opaque = false, valid = true;
};
AffineProgram programTerm(Value v, unsigned depth = 0) {
  if (depth > 64)
    return {0, 0, true, false};
  auto *op = v.getDefiningOp();
  if (!op)
    return {0, 0, true, true};
  auto name = op->getName().getStringRef();
  if (name == "tt.load")
    return {0, 0, true, true};
  if (name == "tt.get_program_id") {
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    return {0, axis && axis.getInt() == 0 ? 1 : 0, false,
            axis && axis.getInt() == 0};
  }
  if (name == "arith.constant") {
    if (auto a = op->getAttrOfType<IntegerAttr>("value")) {
      if (a.getValue().getBitWidth() > 64)
        return {0, 0, true, false};
      return {a.getInt(), 0, false, true};
    }
    if (auto a = op->getAttrOfType<DenseIntElementsAttr>("value"))
      if (a.isSplat()) {
        auto value = a.getSplatValue<APInt>();
        if (value.getBitWidth() > 64)
          return {0, 0, true, false};
        return {value.getSExtValue(), 0, false, true};
      }
    return {0, 0, true, true};
  }
  if (op->getNumOperands() == 1 &&
      llvm::is_contained({StringRef("tt.splat"), StringRef("tt.broadcast"),
                          StringRef("tt.expand_dims"), StringRef("tt.reshape"),
                          StringRef("arith.extsi"), StringRef("arith.extui"),
                          StringRef("arith.index_cast")},
                         name))
    return programTerm(op->getOperand(0), depth + 1);
  if (op->getNumOperands() == 2) {
    auto a = programTerm(op->getOperand(0), depth + 1);
    auto b = programTerm(op->getOperand(1), depth + 1);
    if (!a.valid || !b.valid)
      return {0, 0, true, false};
    if (name == "arith.addi" || name == "tt.addptr" || name == "arith.subi") {
      int64_t k, c;
      bool fail =
          name == "arith.subi"
              ? __builtin_sub_overflow(a.coefficient, b.coefficient, &k) ||
                    __builtin_sub_overflow(a.constant, b.constant, &c)
              : __builtin_add_overflow(a.coefficient, b.coefficient, &k) ||
                    __builtin_add_overflow(a.constant, b.constant, &c);
      return fail ? AffineProgram{0, 0, true, false}
                  : AffineProgram{c, k, a.opaque || b.opaque, true};
    }
    if (name == "arith.muli" && ((!a.opaque && a.coefficient == 0) ||
                                 (!b.opaque && b.coefficient == 0))) {
      if (a.opaque || a.coefficient != 0)
        std::swap(a, b);
      int64_t k, c;
      if (__builtin_mul_overflow(a.constant, b.coefficient, &k) ||
          __builtin_mul_overflow(a.constant, b.constant, &c))
        return {0, 0, true, false};
      return {c, k, b.opaque, true};
    }
    if (a.coefficient || b.coefficient)
      return {0, 0, true, false};
  }
  // Do not silently discard a program term inside an unsupported operation.
  for (Value operand : op->getOperands()) {
    auto a = programTerm(operand, depth + 1);
    if (!a.valid || a.coefficient)
      return {0, 0, true, false};
  }
  return {0, 0, true, true};
}
} // namespace

StageMemoryCost PartialStructuredSimtLoadCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &impl) const {
  StageMemoryCost result;
  if (profile.partialStructuredMemoryModel !=
          "ttir_static_simt_load_factor_20261017" ||
      profile.target != "Ascend950PR/dav-c310" ||
      impl.mode != StageMode::SIMT ||
      (impl.superblockFactor != 2 && impl.superblockFactor != 4) ||
      impl.localScope || impl.aivSubblocks != 1 || stage.features.hasLoop ||
      stage.features.hasAtomicMemory ||
      stage.workload.privateStackLoadInstructions ||
      stage.workload.privateStackStoreInstructions ||
      (stage.costModelKind != StageCostModelKind::PartialContinuousTileMemory &&
       stage.costModelKind != StageCostModelKind::IndirectGatherMemory))
    return result;
  const AddressPatternSummary *target = nullptr;
  for (const auto &pattern : stage.workload.addressPatterns) {
    const bool discrete =
        llvm::any_of(pattern.axes, [](const AddressAxisSummary &a) {
          return a.hasUnknownComponent;
        });
    const bool regular =
        llvm::any_of(pattern.axes, [](const AddressAxisSummary &a) {
          return !a.hasUnknownComponent;
        });
    if (pattern.memoryOp == "tt.store" && discrete)
      return result;
    if (pattern.memoryOp == "tt.load" && discrete && regular) {
      if (target)
        return result;
      target = &pattern;
    }
  }
  if (!target || target->stageId != stage.id)
    return result;
  const auto &p = *target;
  Operation *memory = nullptr;
  bool unsupported = false;
  for (Operation *root : stage.operations)
    root->walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "tt.load" || name == "tt.store") {
        if (name == "tt.load") {
          auto t = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (t && t.getRank() == int64_t(p.axes.size()) &&
              std::equal(t.getShape().begin(), t.getShape().end(),
                         p.axes.begin(),
                         [](int64_t n, const AddressAxisSummary &a) {
                           return n == a.extent;
                         })) {
            if (memory)
              unsupported = true;
            memory = op;
          }
        }
      } else if (!llvm::is_contained({StringRef("tt.splat"),
                                      StringRef("tt.broadcast"),
                                      StringRef("tt.expand_dims"),
                                      StringRef("tt.reshape"),
                                      StringRef("tt.make_range"),
                                      StringRef("tt.addptr"),
                                      StringRef("tt.get_program_id"),
                                      StringRef("tt.bitcast"),
                                      StringRef("arith.constant"),
                                      StringRef("arith.addi"),
                                      StringRef("arith.subi"),
                                      StringRef("arith.muli"),
                                      StringRef("arith.extsi"),
                                      StringRef("arith.extui"),
                                      StringRef("arith.index_cast"),
                                      StringRef("arith.cmpi"),
                                      StringRef("arith.andi"),
                                      StringRef("arith.ori"),
                                      StringRef("arith.divsi"),
                                      StringRef("arith.remsi"),
                                      StringRef("arith.divui"),
                                      StringRef("arith.remui"),
                                      StringRef("scf.if"),
                                      StringRef("scf.yield")},
                                     name))
        unsupported = true;
    });
  if (!memory || unsupported)
    return result;
  auto load = dyn_cast<triton::LoadOp>(memory);
  auto tensor =
      load ? dyn_cast<RankedTensorType>(load.getType()) : RankedTensorType{};
  if (!tensor || !tensor.hasStaticShape() || tensor.getNumElements() > 2048 ||
      p.axes.size() != size_t(tensor.getRank()) ||
      load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL || load.getIsVolatile())
    return result;
  auto program = programTerm(load.getPtr());
  if (!program.valid || program.coefficient <= 0)
    return result;
  partial_memory::FactorLoadGeometry c;
  c.factor = impl.superblockFactor;
  c.programStride = program.coefficient;
  c.geometry.bits = p.elementBitWidth;
  c.geometry.floatingPoint = p.floatingPoint;
  c.geometry.warps = profile.baseSimtWarpCount;
  for (const auto &a : p.axes) {
    const bool d = a.hasUnknownComponent;
    if (d ? (!a.opaqueMultiplier || *a.opaqueMultiplier <= 0 ||
             (a.knownStride && *a.knownStride != 0) ||
             a.regularity != "opaque_loaded")
          : (!a.knownStride || *a.knownStride <= 0))
      return result;
    c.geometry.axes.push_back(
        {a.extent, d ? *a.opaqueMultiplier : *a.knownStride, d});
    c.lower.push_back(0);
    c.upper.push_back(a.extent);
  }
  if (load.getMask()) {
    StaticMask evaluator;
    auto mask = evaluator.evaluate(load.getMask());
    if (!mask || mask->shape != SmallVector<int64_t>(tensor.getShape()))
      return result;
    if (load.getOther()) {
      auto other = evaluator.evaluate(load.getOther());
      // Integer zero tensors are common; floating zero needs an explicit
      // constant check rather than interpreting any runtime value as zero.
      auto *def = load.getOther().getDefiningOp();
      auto floats = def ? def->getAttrOfType<DenseFPElementsAttr>("value")
                        : DenseFPElementsAttr{};
      if ((!other ||
           llvm::any_of(other->values, [](int64_t v) { return v != 0; })) &&
          (!floats ||
           !llvm::all_of(floats.getValues<APFloat>(),
                         [](const APFloat &v) { return v.isZero(); })))
        return result;
    }
    std::fill(c.lower.begin(), c.lower.end(),
              std::numeric_limits<int64_t>::max());
    std::fill(c.upper.begin(), c.upper.end(), 0);
    for (int64_t t = 0; t < tensor.getNumElements(); ++t) {
      if (!mask->values[t])
        continue;
      int64_t flat = t;
      for (size_t i = p.axes.size(); i-- > 0;) {
        const int64_t j = flat % p.axes[i].extent;
        flat /= p.axes[i].extent;
        c.lower[i] = std::min(c.lower[i], j);
        c.upper[i] = std::max(c.upper[i], j + 1);
      }
    }
    // Prove the entire mask is the inferred rectangle, not only its bounding
    // box.
    for (int64_t t = 0; t < tensor.getNumElements(); ++t) {
      int64_t flat = t;
      bool active = true;
      for (size_t i = p.axes.size(); i-- > 0;) {
        int64_t j = flat % p.axes[i].extent;
        flat /= p.axes[i].extent;
        active &= j >= c.lower[i] && j < c.upper[i];
      }
      if (active != bool(mask->values[t]))
        return result;
    }
  }
  auto prediction = partial_memory::predictFactorLoad(c);
  if (!prediction)
    return result;
  StagePartition auxiliary;
  auxiliary.operationOwnershipComplete = true;
  auxiliary.stages.push_back(stage);
  if (auto error =
          StageWorkloadAnalysis().analyzeExcludingMemory(auxiliary, memory)) {
    llvm::consumeError(std::move(error));
    return result;
  }
  result.remainingWorkload = std::move(auxiliary.stages.front().workload);
  result.load = prediction;
  result.replacesTensorMemory = true;
  result.coversSuperblockGroup = true;
  const bool masked =
      c.lower != std::vector<int64_t>(c.lower.size(), 0) ||
      !std::equal(
          c.upper.begin(), c.upper.end(), p.axes.begin(),
          [](int64_t n, const AddressAxisSummary &a) { return n == a.extent; });
  result.pricing = std::string("partial_simt_load_") +
                   (masked ? "mask_" : "unmasked_") + "f" +
                   std::to_string(c.factor) + "_group_formula_20261017";
  return result;
}
