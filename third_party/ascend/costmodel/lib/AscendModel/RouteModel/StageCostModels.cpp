//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double iterations(const LogicalStage &stage) {
  return static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
}

static std::vector<std::string>
collectSourceLocations(const LogicalStage &stage) {
  std::vector<std::string> result;
  llvm::StringSet<> seen;
  for (Operation *operation : stage.operations) {
    std::string location;
    llvm::raw_string_ostream stream(location);
    operation->getLoc().print(stream);
    stream.flush();
    if (location.empty() || !seen.insert(location).second)
      continue;
    result.push_back(std::move(location));
  }
  return result;
}

static double controlBody(const StageResourceCycles &resources) {
  return resources.loopControl + resources.branchControl +
         resources.divergence + resources.synchronization;
}

static double serialBody(const StageResourceCycles &resources) {
  const double execution =
      resources.scalar + resources.load + resources.store + resources.atomic +
      resources.compute + resources.predicate + resources.shuffle +
      resources.dot + controlBody(resources) + resources.spill;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

static bool permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
}

// Scalar white-box terms, already in the profile's SYS_CNT cycle domain.
static double mainScalarLoadCycles(double count,
                                   const StageModeProfile &profile) {
  const double k = std::max(1.0, count);
  const double perLine = k <= profile.mainScalarLoadExtraLineHighThreshold
                             ? profile.mainScalarLoadExtraLineLowCycles
                             : profile.mainScalarLoadExtraLineHighCycles;
  return profile.mainScalarLoadPrepCycles + profile.mainScalarLoadFillCycles +
         std::max(0.0, k - profile.mainScalarLoadOutstandingLines) * perLine +
         (k - 1.0) * profile.mainScalarLoadIssueCycles;
}

static double simtUniformLoadCycles(double count,
                                    const StageModeProfile &profile) {
  return profile.simtUniformLoadPrepCycles + profile.simtUniformLoadFillCycles +
         (std::max(1.0, count) - 1.0) *
             profile.simtUniformLoadDiffLineIssueCycles;
}

static double mte3StoreCycles(const StageModeProfile &profile) {
  return profile.mte3StorePrepCycles + profile.mte3StoreFillCycles;
}

static double simtUniformStoreCycles(const StageModeProfile &profile) {
  return profile.simtUniformStoreBaseCycles;
}

static StageResourceCycles
materializeControlFlow(const LogicalStage &stage, StageMode mode,
                       StageResourceCycles resources,
                       const StageControlFlowRates &rates) {
  resources.loopControl +=
      static_cast<double>(stage.features.loopBackedgeCount) *
      rates.loopBackedgeCycles;
  resources.branchControl +=
      static_cast<double>(stage.features.conditionalBranchCount) *
      rates.conditionalBranchCycles;
  resources.synchronization +=
      static_cast<double>(stage.features.synchronizationCount) *
      rates.synchronizationCycles;
  if (mode == StageMode::SIMT) {
    resources.divergence +=
        static_cast<double>(stage.features.divergentBranchCount) *
        (1.0 - stage.features.activeLaneRatio) *
        rates.divergentBranchPenaltyCycles;
  }
  return resources;
}

// Effective incremental latency of one indirect load (payload - matched ALU),
// not a whole-Stage cost. The random-address prior is intentional: no runtime
// indices or final ISA facts are inputs. Unsupported domains use legacy rates.
static std::optional<double>
calibratedSimdIndirectLoad(const LogicalStage &stage,
                           const HardwareProfile &hardware,
                           const StageImplementation &implementation) {
  const auto &work = stage.workload;
  const bool dtypeRankModel =
      hardware.simdIndirectLoadModel == "random_dtype_matched_ab_20261008";
  if ((!dtypeRankModel &&
       hardware.simdIndirectLoadModel != "random_f32_matched_ab_20261007") ||
      hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMD ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      (!dtypeRankModel && stage.operations.size() != 1) ||
      work.addressPatterns.size() != 1 ||
      work.estimatedSpillTransactions != 0 || work.predicateElements != 0 ||
      stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  // Real Stage partitioning owns the payload's splat/addptr/reshape producers
  // too. Require one load, not one total operation; those independent helpers
  // retain their normal resource charges. Old profiles keep their old guard.
  Operation *op = nullptr;
  for (Operation *candidate : stage.operations) {
    if (candidate->getName().getStringRef() != "tt.load")
      continue;
    if (op)
      return std::nullopt;
    op = candidate;
  }
  if (!op || op->getNumResults() != 1 || op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (dtypeRankModel) {
    if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
        type.getRank() > 5)
      return std::nullopt;
    Type element = type.getElementType();
    const bool supportedInteger =
        element.isInteger(8) || element.isInteger(16) ||
        element.isInteger(32) || element.isInteger(64);
    if (!supportedInteger && !isa<Float16Type, BFloat16Type, Float32Type,
                                  Float8E4M3FNType, Float8E5M2Type>(element))
      return std::nullopt;
    // Triton bool loads are normalized to i8 before this analysis. A raw i1
    // load is not covered: its packed TTIR byte count differs from that ABI.
    const int64_t elements = type.getNumElements();
    const int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
    const auto &pattern = work.addressPatterns.front();
    if (elements < 4 || elements > 2048 || (elements & (elements - 1)) ||
        work.indirectLoadBytes != elementBytes * elements ||
        work.indirectLoadTransactions <= 0 || pattern.memoryOp != "tt.load" ||
        pattern.stageId != stage.id || !pattern.dependsOnLoadedValue ||
        pattern.axes.size() != static_cast<size_t>(type.getRank()))
      return std::nullopt;
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      const auto &summary = pattern.axes[axis];
      const int64_t extent = type.getDimSize(axis);
      if (extent < 2 || (extent & (extent - 1)) || summary.extent != extent)
        return std::nullopt;
      // Reshape can erase per-axis loaded provenance without erasing the
      // whole pointer's loaded-index dependency. The measured inner-index
      // templates produce "opaque" here; do not reject them as direct loads.
      const bool opaque = summary.regularity == "opaque_loaded" ||
                          summary.regularity == "opaque";
      const bool fixedOuter = axis + 1 < type.getRank() &&
                              summary.regularity == "fixed_stride" &&
                              summary.knownStride > 0;
      if (!opaque && !fixedOuter)
        return std::nullopt;
    }
    if (type.getRank() == 2 &&
        (elements < 32 || type.getDimSize(1) < 4 || type.getDimSize(1) > 128))
      return std::nullopt;
    // Effective matched-address load increment in SYS_CNT/system_cycles.
    // The tiny 32-bit path has measured unrolling/grouping differences; it is
    // not a universal hardware latency threshold. No SIMD warp division.
    const bool small32 =
        elementBytes == 4 && type.getRank() == 1 && elements <= 16;
    return elements * (small32 ? 49.57621548794853 : 81.94869549595556);
  }
  // Keep the previously versioned FP32 model reproducible for old profiles.
  if (!type || !type.hasStaticShape() ||
      (type.getRank() != 1 && type.getRank() != 2) ||
      !type.getElementType().isF32())
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  if (pattern.memoryOp != "tt.load" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      pattern.axes.back().regularity != "opaque_loaded" || elements < 8 ||
      elements > 512 || (elements & (elements - 1)) ||
      work.indirectLoadBytes != 4 * elements ||
      work.indirectLoadTransactions <= 0)
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (pattern.axes[axis].extent != type.getDimSize(axis))
      return std::nullopt;
  if (type.getRank() == 2) {
    const auto &outer = pattern.axes.front();
    const int64_t columns = type.getDimSize(1);
    // Rank-two validation currently supports the narrow-column domain only.
    // Wider columns have measured counterexamples; keep their legacy fallback.
    if (elements < 32 || (columns != 4 && columns != 8) ||
        (outer.regularity != "opaque_loaded" &&
         !(outer.regularity == "fixed_stride" && outer.knownStride == 4096)))
      return std::nullopt;
  }
  // Effective load increment, NOT the old 61.72 + 96.52*N whole-loop fit.
  // The address-preserving baseline has a small extra ALU bias; random-only
  // input validation and baseline-sensitivity results accompany calibration.
  return 83.56748010753823 * elements;
}

static std::optional<double>
calibratedSimtDtypeIndirectLoad(const LogicalStage &stage,
                                const HardwareProfile &hardware,
                                const StageImplementation &implementation) {
  const auto &work = stage.workload;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      work.addressPatterns.size() != 1 ||
      work.estimatedSpillTransactions != 0 || work.predicateElements != 0 ||
      stage.features.hasAtomicMemory || stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  // Partitioned Stages also own address/shape producers. Their resources stay
  // independent; the fit replaces only the single target indirect load.
  // Explicit SIMT regions may be represented by their owning scope op. Walk
  // owned regions too; counting only top-level ops misses real Triton loads.
  llvm::SmallPtrSet<Operation *, 4> loads;
  for (Operation *owned : stage.operations)
    owned->walk([&](Operation *candidate) {
      if (candidate->getName().getStringRef() == "tt.load")
        loads.insert(candidate);
    });
  if (loads.size() != 1)
    return std::nullopt;
  Operation *op = *loads.begin();
  if (op->getNumResults() != 1 || op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 5)
    return std::nullopt;
  Type element = type.getElementType();
  const bool supportedInteger = element.isInteger(8) || element.isInteger(16) ||
                                element.isInteger(32) || element.isInteger(64);
  if (!supportedInteger &&
      !isa<Float16Type, BFloat16Type, Float32Type, Float64Type,
           Float8E4M3FNType, Float8E5M2Type>(element))
    return std::nullopt;
  // Raw i1 is not a one-byte TTIR load; normalized bool loads use i8.
  const int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  const auto &pattern = work.addressPatterns.front();
  if (pattern.memoryOp != "tt.load" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()))
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    const auto &summary = pattern.axes[axis];
    const int64_t extent = type.getDimSize(axis);
    const bool opaque =
        summary.regularity == "opaque_loaded" || summary.regularity == "opaque";
    const bool fixedOuter = axis + 1 < type.getRank() &&
                            summary.regularity == "fixed_stride" &&
                            summary.knownStride > 0;
    if (extent < 2 || (extent & (extent - 1)) || summary.extent != extent ||
        (!opaque && !fixedOuter))
      return std::nullopt;
  }
  const int64_t warps = hardware.logicalWarpGroupCount;
  if (warps < 1 || warps > 64 || (warps & (warps - 1)))
    return std::nullopt;
  const double elements = type.getNumElements();
  const double q = elements / (32.0 * warps);
  const int64_t columns = type.getRank() == 1 ? 32 : type.getShape().back();
  if ((q != 1 && q != 2 && q != 4) || columns < 2 || columns > 128 ||
      work.indirectLoadBytes != elementBytes * elements ||
      work.indirectLoadTransactions <= 0)
    return std::nullopt;
  const double h = std::max(4 * q / columns - 2, 0.0);
  // C=2,q=4 has H=6, beyond the calibrated H<=2 range and with measured
  // counterexamples. Do not silently extrapolate this baseline correction.
  if (h > 2)
    return std::nullopt;
  const double d = std::max(std::log2(32.0 / columns), 0.0);
  struct Coefficients {
    double fixed, elements, small, flat, rowBaseline, narrowBaseline;
  };
  // Frozen pure-random, four-storage-width calibration. INT32/UINT32/FP32
  // share the 4-byte row; dtype names do not select historical coefficients.
  static constexpr Coefficients coefficients[] = {
      {7.617911783542814, 1.5780971099377332, 92.18952965944916,
       0.19387737354356369, 0.0, 0.049471659398567715},
      {12.456944150614481, 1.6732132663368477, 95.13859585355112,
       0.11214618741568584, 5.634318857353958, 0.026588037490546692},
      {50.71936539506111, 1.693396365504775, 60.38681262899275,
       0.09265123974451413, 82.91193957489877, 0.026814982781878355},
      {96.09572807125711, 1.937402636239715, 24.962585867214496,
       0.20099293498091206, 41.8669299352394, 0.03855485734655757}};
  const auto &c = coefficients[elementBytes == 1   ? 0
                               : elementBytes == 2 ? 1
                               : elementBytes == 4 ? 2
                                                   : 3];
  // Effective A/B increment in SYS_CNT/system_cycles, already including W
  // warp parallelism. H and E*D compensate baseline mismatch, not GM latency.
  return c.fixed + c.elements * elements + c.small * (elements <= 128) +
         c.flat * elements * (type.getRank() == 1) + c.rowBaseline * h +
         c.narrowBaseline * elements * d;
}

static std::optional<double>
calibratedIndirectLoad(const LogicalStage &stage,
                       const HardwareProfile &hardware,
                       const StageImplementation &implementation) {
  const auto &work = stage.workload;
  if (implementation.mode == StageMode::SIMD)
    return calibratedSimdIndirectLoad(stage, hardware, implementation);
  if (hardware.simtIndirectLoadModel == "random_dtype_six_term_20261008")
    return calibratedSimtDtypeIndirectLoad(stage, hardware, implementation);
  // Explicitly versioned compatibility model, not an INT32 exception inside
  // the four-width model. Old profiles retain their coefficients and guards.
  if (hardware.simtIndirectLoadModel != "random_i32_six_term_20261007" ||
      hardware.target != "Ascend950PR/dav-c310" ||
      implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      stage.operations.size() != 1 || work.addressPatterns.size() != 1 ||
      work.estimatedSpillTransactions != 0 || work.predicateElements != 0 ||
      stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousLoadBytes != 0 || work.indirectStoreBytes != 0)
    return std::nullopt;
  Operation *op = stage.operations.front();
  if (op->getName().getStringRef() != "tt.load" || op->getNumResults() != 1 ||
      op->getNumOperands() != 1)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isInteger(32) ||
      (type.getRank() != 1 && type.getRank() != 2))
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  if (pattern.memoryOp != "tt.load" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()))
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (pattern.axes[axis].extent != type.getDimSize(axis))
      return std::nullopt;
  if (pattern.axes.back().regularity != "opaque_loaded")
    return std::nullopt;
  if (type.getRank() == 2 &&
      pattern.axes.front().regularity != "opaque_loaded" &&
      !(pattern.axes.front().regularity == "fixed_stride" &&
        pattern.axes.front().knownStride == 4096))
    return std::nullopt;
  const int64_t warps = hardware.logicalWarpGroupCount;
  if (warps < 1 || warps > 64 || (warps & (warps - 1)))
    return std::nullopt;
  const double elements = type.getNumElements();
  const double q = elements / (32.0 * warps);
  const int64_t columns = type.getRank() == 1 ? 32 : type.getDimSize(1);
  if ((q != 1 && q != 2 && q != 4) ||
      (type.getRank() == 2 && type.getDimSize(0) < 2) || columns < 4 ||
      columns > 128 || (columns & (columns - 1)) ||
      work.indirectLoadBytes != 4 * elements ||
      work.indirectLoadTransactions <= 0)
    return std::nullopt;
  const double h = std::max(4 * q / columns - 2, 0.0);
  const double d = std::max(std::log2(32.0 / columns), 0.0);
  // SYS_CNT ticks are already CostModel system_cycles (no clock conversion).
  return 62.930686069640686 + 1.5502588407166296 * elements +
         141.51064147799983 * h + 47.159408679724294 * (elements <= 128) +
         0.05304437217224805 * elements * d +
         0.19338028618547126 * elements * (type.getRank() == 1);
}

// Random, collision-free addresses; no target fill. Coefficients price the
// measured A/B increment in SYS_CNT ticks, not a worst-case latency bound.
static std::optional<double>
randomDtypeIndirectStore(const LogicalStage &stage,
                         const HardwareProfile &hardware, bool simd,
                         bool reuse) {
  const auto &work = stage.workload;
  if (reuse && !work.hasProvenIndirectStoreReuse)
    return std::nullopt;
  // Real partitioning also owns the store's address producers. Require one
  // target store, not one total operation; helpers keep their normal charges.
  Operation *op = nullptr;
  for (Operation *candidate : stage.operations) {
    if (candidate->getName().getStringRef() != "tt.store")
      continue;
    if (op)
      return std::nullopt;
    op = candidate;
  }
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 0)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!type || !type.hasStaticShape() || type.getRank() < 1 ||
      type.getRank() > 8)
    return std::nullopt;
  Type element = type.getElementType();
  if (!isa<IntegerType>(element) && !element.isF16() && !element.isBF16() &&
      !element.isF32() && !(element.isF64() && !simd) &&
      !isa<Float8E4M3FNType, Float8E5M2Type>(element))
    return std::nullopt;
  unsigned bits = element.getIntOrFloatBitWidth();
  if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  const int64_t warps = hardware.logicalWarpGroupCount;
  if (elements < 8 || elements > (simd ? 4096 : 16384) ||
      (elements & (elements - 1)) || warps < 1 || warps > 64 ||
      (warps & (warps - 1)) || (simd && warps != 1) ||
      (!simd &&
       (elements > 512 * warps || (warps > 1 && elements < 32 * warps))) ||
      pattern.memoryOp != "tt.store" || pattern.stageId != stage.id ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      work.indirectStoreBytes != std::max(1u, bits / 8) * elements ||
      work.indirectStoreTransactions <= 0)
    return std::nullopt;
  // Every axis must itself prove loaded provenance. This is stronger than
  // the legacy whole-pointer dependency predicate, whose rank cap excludes
  // ranks 6-8; do not use that predicate to reject proven high-rank axes.
  // The calibrated prior excludes broadcast/structured-prefix addresses.
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    int64_t extent = type.getDimSize(axis);
    if (extent < 1 || (extent & (extent - 1)) ||
        pattern.axes[axis].extent != extent ||
        pattern.axes[axis].regularity != "opaque_loaded")
      return std::nullopt;
  }
  const unsigned group = bits <= 16 ? 0 : (bits == 32 ? 1 : 2);
  if (simd) {
    const double first[] = {152.8758408085307, 162.87111975882544,
                            161.07674811788297};
    const double warm[] = {68.58965840703893, 69.21236829277788,
                           64.45492494749357};
    return (reuse ? 0.0 : 125.41805018354371) +
           (reuse ? warm[group] : first[group]) * elements;
  }
  const double firstSlope[] = {4.664669494263474, 1.3935292896269613,
                               2.0019961511928295};
  const double warmSlope[] = {4.4638287665056, 1.1754145133075695,
                              2.076648173015399};
  const double firstWarp[] = {-25.526112727120946, -19.191779247532597,
                              -35.604138981973364};
  const double warmWarp[] = {-9.072748388478542, -20.88472709201174,
                             -38.4225252759396};
  // q is a fitted scheduling correction, not a negative hardware latency.
  return (reuse ? 70.0956884939018 : 382.1012717872757) +
         (reuse ? warmSlope[group] : firstSlope[group]) * elements +
         (reuse ? 0.8774560851959086 : 0.6797627278581438) *
             std::max<int64_t>(elements - (reuse ? 128 : 256), 0) +
         (reuse ? warmWarp[group] : firstWarp[group]) *
             (elements / (32.0 * warps));
}

// Effective write plus required synchronization increment, not a whole Stage.
// Address randomness and memory history are profile priors: neither is proven
// by pointer analysis. In particular, the no-fill first fit has >20% outliers.
static std::optional<double>
calibratedIndirectStore(const LogicalStage &stage,
                        const HardwareProfile &hardware,
                        const StageImplementation &implementation) {
  const auto &work = stage.workload;
  const bool simd = implementation.mode == StageMode::SIMD;
  if (hardware.target != "Ascend950PR/dav-c310" ||
      implementation.superblockFactor != 1 ||
      stage.costModelKind != StageCostModelKind::IndirectGatherMemory ||
      work.addressPatterns.size() != 1 ||
      work.estimatedSpillTransactions != 0 || work.predicateElements != 0 ||
      stage.features.hasAtomicMemory || stage.features.activeLaneRatio != 1.0 ||
      work.partialContinuousStoreBytes != 0 || work.indirectLoadBytes != 0)
    return std::nullopt;
  const auto &model =
      simd ? hardware.simdIndirectStoreModel : hardware.simtIndirectStoreModel;
  if ((simd || implementation.mode == StageMode::SIMT) &&
      (model == "random_store_no_fill_first_20261008" ||
       model == "random_store_no_fill_reuse_20261008"))
    return randomDtypeIndirectStore(
        stage, hardware, simd, model == "random_store_no_fill_reuse_20261008");
  // Preserve the historical FP32 model's original admission boundary.
  if (stage.operations.size() != 1)
    return std::nullopt;
  if (simd) {
    if ((hardware.simdIndirectStoreModel !=
             "random_f32_store_no_fill_first_20261007" &&
         hardware.simdIndirectStoreModel !=
             "random_f32_store_no_fill_reuse_20261007") ||
        hardware.logicalWarpGroupCount != 1)
      return std::nullopt;
  } else if (implementation.mode != StageMode::SIMT ||
             hardware.simtIndirectStoreModel !=
                 "random_f32_store_fill_ab_20261007") {
    return std::nullopt;
  }
  Operation *op = stage.operations.front();
  if (op->getName().getStringRef() != "tt.store" || op->getNumOperands() != 2 ||
      op->getNumResults() != 0)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      (type.getRank() != 1 && type.getRank() != 2))
    return std::nullopt;
  const auto &pattern = work.addressPatterns.front();
  const int64_t elements = type.getNumElements();
  if (pattern.memoryOp != "tt.store" || pattern.stageId != stage.id ||
      !pattern.dependsOnLoadedValue ||
      pattern.axes.size() != static_cast<size_t>(type.getRank()) ||
      pattern.axes.back().regularity != "opaque_loaded" ||
      work.indirectStoreBytes != 4 * elements ||
      work.indirectStoreTransactions <= 0)
    return std::nullopt;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (pattern.axes[axis].extent != type.getDimSize(axis))
      return std::nullopt;
  const bool rankTwo = type.getRank() == 2;
  const int64_t columns = rankTwo ? type.getDimSize(1) : 32;
  const bool inner =
      rankTwo && pattern.axes.front().regularity == "fixed_stride";
  if (rankTwo &&
      (type.getDimSize(0) < 2 ||
       (pattern.axes.front().regularity != "opaque_loaded" && !inner) ||
       (inner && pattern.axes.front().knownStride != (simd ? 32768 : 4096))))
    return std::nullopt;
  if (simd) {
    // Exactly the measured shape domain. Row/column loaded and per-element
    // loaded templates share this slope; unsupported expressions fall back.
    if ((!rankTwo &&
         (elements < 8 || elements > 512 || (elements & (elements - 1)))) ||
        (rankTwo && ((elements != 32 && elements != 128 && elements != 512) ||
                     (columns != 4 && columns != 16 && columns != 64))))
      return std::nullopt;
    if (hardware.simdIndirectStoreModel ==
            "random_f32_store_no_fill_reuse_20261007" &&
        !work.hasProvenIndirectStoreReuse)
      return std::nullopt;
    return elements * (hardware.simdIndirectStoreModel ==
                               "random_f32_store_no_fill_reuse_20261007"
                           ? 65.26772542192847
                           : 150.13769870695648);
  }
  const int64_t warps = hardware.logicalWarpGroupCount;
  if (warps < 1 || warps > 64 || (warps & (warps - 1)))
    return std::nullopt;
  const double q = elements / (32.0 * warps);
  if ((q != 1 && q != 2 && q != 4) ||
      (rankTwo && columns != 8 && columns != 32))
    return std::nullopt;
  if (elements >= 512)
    return 1.9039194506414692 * elements;
  const double logW = std::log2(static_cast<double>(warps));
  return 157.465336398077 + 2.046486341690834 * elements -
         13.874579575172818 * std::sqrt(static_cast<double>(elements)) +
         (-0.8540920211063882 - 0.4681840422126738 * q) * logW +
         0.5780871675460912 * elements / columns - 15.490435818576909 * inner;
}

static StageResourceCycles mapWorkload(const LogicalStage &stage,
                                       const StageModeProfile &profile,
                                       StageMode mode,
                                       std::optional<double> fittedLoad,
                                       std::optional<double> fittedStore) {
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  const bool simd = mode == StageMode::SIMD;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  llvm::StringMap<double> describedElements;
  llvm::StringMap<double> describedVectorInstructions;
  double describedIssueElements = 0.0;
  double describedIssueInstructions = 0.0;
  if (simd) {
    for (const TensorOperationWorkload &tensor :
         work.tensorOperationWorkloads) {
      describedElements[tensor.operation] += tensor.logicalElements;
      describedIssueElements += tensor.logicalElements;
      const double segmentBits =
          static_cast<double>(tensor.contiguousElementsPerSegment) *
          static_cast<double>(tensor.elementBitWidth);
      const double vectorInstructions =
          tensor.segmentCount *
          std::ceil(segmentBits / static_cast<double>(profile.vectorWidthBits));
      describedVectorInstructions[tensor.operation] += vectorInstructions;
      describedIssueInstructions += vectorInstructions;
    }
  }
  for (const auto &[name, elements] : work.operationElements) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    double instructions = elements;
    if (simd) {
      const double described = describedElements.lookup(name);
      const double tolerance = 1e-9 * std::max(1.0, elements);
      instructions =
          std::abs(described - elements) <= tolerance
              ? describedVectorInstructions.lookup(name)
              : std::ceil(elements / static_cast<double>(profile.vectorWidth));
    }
    resources.compute +=
        instructions / rate->second.throughput * rate->second.factor;
  }
  resources.scalar += work.scalarOperations / profile.scalarOperationsPerCycle;
  const double directLoadBytes = work.loadBytes - work.indirectLoadBytes;
  const double directStoreBytes = work.storeBytes - work.indirectStoreBytes;
  const double directLoadInstructions =
      work.loadWarpInstructions - work.indirectLoadTransactions;
  const double directStoreInstructions =
      work.storeWarpInstructions - work.indirectStoreTransactions;
  if (simd) {
    resources.load = directLoadBytes / profile.loadBytesPerCycle;
    resources.store = directStoreBytes / profile.storeBytesPerCycle;
  } else {
    resources.load =
        directLoadInstructions / profile.loadWarpInstructionsPerCycle;
    resources.store =
        directStoreInstructions / profile.storeWarpInstructionsPerCycle;
  }
  resources.load += fittedLoad.value_or(
      work.indirectLoadTransactions / profile.indirectLoadTransactionsPerCycle);
  resources.store +=
      fittedStore.value_or(work.indirectStoreTransactions /
                           profile.indirectStoreTransactionsPerCycle);
  // Preserve one uncovered loaded-index dependency latency per Stage
  // iteration, but charge it only when an actual indirect access exists.
  if (work.indirectLoadTransactions > 0.0) {
    // Matched-load difference already includes the induced index wait.
    if (!fittedLoad)
      resources.load += profile.indirectDependencyLatencyCycles;
  } else if (work.indirectStoreTransactions > 0.0 && !fittedStore)
    // The matched-store increment replaces the legacy write/wait estimate.
    resources.store += profile.indirectDependencyLatencyCycles;

  for (const AtomicWorkload &atomic : work.atomicWorkloads) {
    auto rate = profile.atomicRates.find(atomic.profileKey());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find("default");
    if (rate == profile.atomicRates.end())
      continue;
    const StageAtomicRate &atomicRate = rate->second;
    const double base =
        atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
        atomic.logicalElements / atomicRate.logicalElementsPerCycle;
    const double contention =
        atomic.contentionUnknown ? atomicRate.unknownContentionMultiplier : 1.0;
    resources.atomic += base * contention;
    if (atomic.resultUsed)
      resources.atomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
  }
  if (work.scalarLoadCount > 0.0) {
    resources.load +=
        simd ? mainScalarLoadCycles(work.scalarLoadCount, profile)
             : simtUniformLoadCycles(work.scalarLoadCount, profile);
  }
  if (work.scalarStoreCount > 0.0) {
    resources.store +=
        simd ? mte3StoreCycles(profile) : simtUniformStoreCycles(profile);
  }
  double predicateInstructions = work.predicateElements;
  if (simd) {
    const double described = describedElements.lookup("predicate.cmp");
    const double tolerance = 1e-9 * std::max(1.0, work.predicateElements);
    predicateInstructions =
        std::abs(described - work.predicateElements) <= tolerance
            ? describedVectorInstructions.lookup("predicate.cmp")
            : std::ceil(work.predicateElements /
                        static_cast<double>(profile.vectorWidth));
  }
  resources.predicate =
      predicateInstructions / profile.predicateOperationsPerCycle;
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  resources.scanShuffle =
      work.scanShuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (work.dotFlops > 0.0) {
    resources.setup += profile.dotSetupCycles;
    resources.dot = work.dotFlops / profile.dotFlopsPerCycle;
  }
  double issueInstructions =
      std::ceil(work.issueElements / static_cast<double>(profile.issueWidth));
  if (simd) {
    // Keep the issue floor consistent with operation pricing.  Aggregating
    // unrelated short rows before dividing by the vector width makes several
    // independently issued instructions look like one full-width operation.
    const double undescribedIssueElements =
        std::max(0.0, work.issueElements - describedIssueElements);
    issueInstructions = describedIssueInstructions +
                        std::ceil(undescribedIssueElements /
                                  static_cast<double>(profile.issueWidth));
  }
  resources.issue = issueInstructions / profile.issueOperationsPerCycle;
  resources.spill =
      work.estimatedSpillTransactions / profile.spillTransactionsPerCycle;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.shuffle +
                             resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath =
        resources.compute + resources.predicate + resources.shuffle;
  return materializeControlFlow(stage, mode, resources, profile.controlFlow);
}

static double applySuperBlock(const LogicalStage &stage,
                              const StageResourceCycles &resources,
                              const StageImplementation &implementation,
                              const HardwareProfile &profile,
                              double stageCycles) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;

  const double factor = static_cast<double>(implementation.superblockFactor);
  const double effectiveFactor = std::min(
      factor, static_cast<double>(profile.superblockUsefulFactorLimit));
  const double latencySensitivePerIteration =
      resources.load + resources.store + resources.atomic + resources.shuffle +
      resources.divergence;
  const double latencySensitive =
      iterations(stage) * latencySensitivePerIteration;
  // SuperBlock creates `factor` independent logical-program groups on one
  // physical core.  It can hide latency across those groups, but it cannot
  // divide dependent arithmetic, loop control, or synchronization.
  const double pressure =
      iterations(stage) * resources.spill * std::max(0.0, factor - 1.0);
  // Live-out bytes alone do not prove register pressure: they describe the
  // Stage ABI, not the allocator's simultaneously-live set.  Charge replicated
  // persistent state only when workload analysis has independently predicted
  // spill traffic.  This keeps the penalty evidence based and lets independent
  // recurrence groups use F4 when the generated SIMT VF has no STK/LDK.
  const double persistentStatePressure =
      stage.features.hasLoopCarriedDataDependency && resources.spill > 0.0
          ? std::max(
                0.0,
                factor -
                    static_cast<double>(
                        profile.superblockPersistentStatePressureFreeFactor)) *
                static_cast<double>(stage.liveOutBytes) /
                profile.superblockPersistentStateBytesPerCycle
          : 0.0;
  const double fixed = resources.setup;
  const double issueFloor =
      fixed + factor * iterations(stage) * resources.issue;
  // A recurrence is serial inside one logical program.  SuperBlock contributes
  // F independent logical programs to the same physical program, allowing the
  // scheduler to cover one program's dependency stalls with another program.
  // Normalize the critical-path portion per logical program, but retain the
  // aggregate issue floor: a larger factor cannot create additional issue
  // bandwidth.
  // This applies equally to whole-kernel and scope-local SuperBlock because
  // both materializers batch complete logical programs around the Stage.
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence) {
    const double recurrenceBody = std::max(0.0, stageCycles - fixed);
    return std::max(issueFloor, fixed + recurrenceBody + pressure) +
           persistentStatePressure;
  }
  // Proven persistent-state pressure is additional register/stack work and
  // cannot disappear behind the ordinary issue floor.
  const double body = std::max(0.0, stageCycles - fixed);
  const double groupedBody = factor * std::max(0.0, body - latencySensitive) +
                             factor * latencySensitive / effectiveFactor;
  return std::max(issueFloor, fixed + groupedBody + pressure) +
         persistentStatePressure;
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            const StageResourceCycles &r) {
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  switch (stage.costModelKind) {
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop: {
    const double dispatchCount =
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ? count
                                                                    : 1.0;
    return r.setup +
           dispatchCount * std::max(r.scalar + controlBody(r), r.issue);
  }
  case StageCostModelKind::PartialContinuousTileMemory:
    // Slices expanded along unstructured axes are billed serially. Structured
    // axes may have nonunit strides; they do not imply contiguous addresses.
    return serial;
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
  case StageCostModelKind::AtomicMemory:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count * (r.scalar + r.predicate + controlBody(r) + r.spill +
                      std::max({r.load, r.store, r.atomic, r.issue}));
    return serial;
  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count *
                 (std::max({r.load, r.store, r.atomic,
                            r.compute + r.dot + r.shuffle,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    return serial;
  case StageCostModelKind::LoopCarriedRecurrence: {
    // A prefix scan nested inside a recurrence keeps its lane dependency
    // chain: each scan level must complete before the next starts, so the
    // scan's shuffle traffic cannot reach the ideal vector throughput.
    // Scale only the tt.scan-contributed portion of the shuffle critical
    // path with the same mode-specific dependency factor the standalone
    // PrefixScan model uses (identity for SIMT).  tt.reduce-contributed
    // shuffle keeps the ideal rate: tree reductions halve their active
    // lanes per level, so the N*log2(N) lane-step billing already carries
    // enough slack to absorb per-level inefficiency.
    double critical = r.criticalPath > 0.0
                          ? std::max(r.criticalPath + r.load + r.store +
                                         r.atomic + controlBody(r) + r.spill,
                                     r.issue)
                          : serialBody(r);
    if (r.criticalPath > 0.0 && stage.features.hasPrefixScan) {
      const double dependencyFactor =
          mode == StageMode::SIMD ? profile.simd.prefixScanDependencyFactor
                                  : profile.simt.prefixScanDependencyFactor;
      critical =
          std::max(r.criticalPath + (dependencyFactor - 1.0) * r.scanShuffle +
                       r.load + r.store + r.atomic + controlBody(r) + r.spill,
                   r.issue);
    }
    if (mode == StageMode::SIMD) {
      // A loop-carried tensor is not ordinary embarrassingly-parallel vector
      // work: the updated state must remain live until the next recurrence
      // step.  The operation-throughput terms above account for arithmetic,
      // but not this persistent register/stack traffic.  Charge the exact
      // SSA live-out footprint once per Stage invocation using the target
      // profile's persistent-state byte rate.
      const double persistentState =
          static_cast<double>(stage.liveOutBytes) /
          profile.superblockPersistentStateBytesPerCycle;
      return r.setup + count * critical + persistentState;
    }
    const int64_t groups = std::max<int64_t>(
        1, std::min(stage.features.parallelRecurrenceGroupCount,
                    profile.logicalWarpGroupCount));
    return r.setup +
           std::max(std::ceil(count / static_cast<double>(groups)) * critical,
                    count * r.issue);
  }
  case StageCostModelKind::RowwiseReduction:
    return r.setup +
           count * std::max(r.scalar + r.load + r.store + r.atomic +
                                r.criticalPath + controlBody(r) + r.spill,
                            r.issue);
  case StageCostModelKind::PrefixScan: {
    const double scanCritical =
        r.compute + r.predicate +
        r.shuffle * (mode == StageMode::SIMD
                         ? profile.simd.prefixScanDependencyFactor
                         : profile.simt.prefixScanDependencyFactor);
    return r.setup +
           count * std::max(r.scalar + r.load + r.store + r.atomic +
                                scanCritical + controlBody(r) + r.spill,
                            r.issue);
  }
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup + count * (r.scalar + r.predicate + controlBody(r) +
                                r.shuffle + r.spill +
                                std::max({r.load, r.compute + r.dot, r.store,
                                          r.atomic, r.issue}));
    return serial;
  case StageCostModelKind::ConversionPack:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup + count * (r.predicate + controlBody(r) + r.spill +
                                std::max({r.scalar + r.compute, r.load, r.store,
                                          r.atomic, r.issue}));
    return serial;
  default:
    if (mode == StageMode::SIMD)
      return r.setup +
             count *
                 (std::max({r.load, r.store, r.atomic,
                            r.compute + r.dot + r.shuffle,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    return serial;
  }
}
static bool isDeclaredLegal(const LogicalStage &stage,
                            const StageImplementation &implementation) {
  if (!implementation.isValid())
    return false;
  if (implementation.mode == StageMode::SIMD)
    return stage.simdLegal && implementation.superblockFactor == 1 &&
           !implementation.localScope;
  if (!stage.simtLegal)
    return false;
  if (implementation.localScope)
    return stage.localSimtMaterializable &&
           llvm::is_contained(stage.localSimtFactors,
                              implementation.superblockFactor);
  return llvm::is_contained(stage.legalSimtFactors,
                            implementation.superblockFactor);
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageCostModel(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
    return "auto_blockify_dispatch";
  case StageCostModelKind::AutoBlockifyLoop:
    return "auto_blockify_loop";
  case StageCostModelKind::ScalarIssue:
    return "scalar_issue";
  case StageCostModelKind::ScalarControl:
    return "scalar_control";
  case StageCostModelKind::ScalarMath:
    return "scalar_math";
  case StageCostModelKind::ScalarLoad:
    return "scalar_load";
  case StageCostModelKind::ScalarStore:
    return "scalar_store";
  case StageCostModelKind::IndexGeneration:
    return "index_generation";
  case StageCostModelKind::PredicateMask:
    return "predicate_mask";
  case StageCostModelKind::LoopPredicate:
    return "loop_predicate";
  case StageCostModelKind::ContinuousTileMemory:
    return "continuous_tile_memory";
  case StageCostModelKind::PartialContinuousTileMemory:
    return "partial_continuous_tile_memory";
  case StageCostModelKind::ContinuousTileStore:
    return "continuous_tile_store";
  case StageCostModelKind::ContinuousShortLoad:
    return "continuous_short_load";
  case StageCostModelKind::CachePolicyStore:
    return "cache_policy_store";
  case StageCostModelKind::IndirectScalarMemory:
    return "indirect_scalar_memory";
  case StageCostModelKind::IndirectGatherMemory:
    return "indirect_gather_memory";
  case StageCostModelKind::AtomicMemory:
    return "atomic_memory";
  case StageCostModelKind::IndependentPipelinedLoop:
    return "independent_pipelined_loop";
  case StageCostModelKind::LoopCarriedRecurrence:
    return "loop_carried_recurrence";
  case StageCostModelKind::RowwiseReduction:
    return "rowwise_reduction";
  case StageCostModelKind::PrefixScan:
    return "prefix_scan";
  case StageCostModelKind::CubeRoofline:
    return "cube_roofline";
  case StageCostModelKind::TinyCubeRoofline:
    return "tiny_cube_roofline";
  case StageCostModelKind::ConversionPack:
    return "conversion_pack";
  }
  llvm_unreachable("unknown StageCostModelKind");
}

bool StageControlFlowRates::isFiniteAndNonNegative() const {
  const std::array<double, 4> values = {
      loopBackedgeCycles, conditionalBranchCycles, divergentBranchPenaltyCycles,
      synchronizationCycles};
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

bool StageAtomicRate::isValid() const {
  return std::isfinite(logicalElementsPerCycle) &&
         logicalElementsPerCycle > 0.0 &&
         std::isfinite(operationStartupCycles) &&
         operationStartupCycles >= 0.0 &&
         std::isfinite(resultDependencyCycles) &&
         resultDependencyCycles >= 0.0 &&
         std::isfinite(unknownContentionMultiplier) &&
         unknownContentionMultiplier >= 1.0;
}

bool StageModeProfile::isValid(StageMode mode) const {
  const std::array<double, 14> common = {setupCycles,
                                         predicateOperationsPerCycle,
                                         shuffleLanesPerCycle,
                                         dotSetupCycles,
                                         dotFlopsPerCycle,
                                         scalarOperationsPerCycle,
                                         issueOperationsPerCycle,
                                         spillTransactionsPerCycle,
                                         indirectLoadTransactionsPerCycle,
                                         indirectStoreTransactionsPerCycle,
                                         prefixScanDependencyFactor,
                                         static_cast<double>(vectorWidthBits),
                                         static_cast<double>(vectorWidth),
                                         static_cast<double>(issueWidth)};
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(indirectDependencyLatencyCycles) ||
      indirectDependencyLatencyCycles < 0.0 ||
      !controlFlow.isFiniteAndNonNegative())
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  return llvm::all_of(operationRates,
                      [](const auto &entry) {
                        return std::isfinite(entry.second.throughput) &&
                               entry.second.throughput > 0.0 &&
                               std::isfinite(entry.second.factor) &&
                               entry.second.factor > 0.0;
                      }) &&
         atomicRates.contains("default") &&
         llvm::all_of(atomicRates,
                      [](const auto &entry) { return entry.second.isValid(); });
}

bool HardwareProfile::isValid() const {
  return !profileVersion.empty() && !target.empty() &&
         (simtIndirectLoadModel.empty() ||
          ((simtIndirectLoadModel == "random_i32_six_term_20261007" ||
            simtIndirectLoadModel == "random_dtype_six_term_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         (simdIndirectLoadModel.empty() ||
          ((simdIndirectLoadModel == "random_f32_matched_ab_20261007" ||
            simdIndirectLoadModel == "random_dtype_matched_ab_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         (simtIndirectStoreModel.empty() ||
          ((simtIndirectStoreModel == "random_f32_store_fill_ab_20261007" ||
            simtIndirectStoreModel == "random_store_no_fill_first_20261008" ||
            simtIndirectStoreModel == "random_store_no_fill_reuse_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         (simdIndirectStoreModel.empty() ||
          ((simdIndirectStoreModel ==
                "random_f32_store_no_fill_first_20261007" ||
            simdIndirectStoreModel ==
                "random_f32_store_no_fill_reuse_20261007" ||
            simdIndirectStoreModel == "random_store_no_fill_first_20261008" ||
            simdIndirectStoreModel == "random_store_no_fill_reuse_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         logicalWarpGroupCount > 0 && superblockUsefulFactorLimit > 0 &&
         superblockPersistentStatePressureFreeFactor > 0 &&
         superblockPersistentStatePressureFreeFactor <=
             superblockUsefulFactorLimit &&
         std::isfinite(superblockPersistentStateBytesPerCycle) &&
         superblockPersistentStateBytesPerCycle > 0.0 &&
         simd.isValid(StageMode::SIMD) && simt.isValid(StageMode::SIMT) &&
         transition.isValid();
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StagePartition &partition,
                             const HardwareProfile &profile) const {
  if (partition.stages.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires at least one Stage");
  if (!profile.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "HardwareProfile is invalid");
  StageCostTable table;
  table.operationOwnershipComplete = partition.operationOwnershipComplete;
  table.modeledOperationCount = partition.modeledOperationCount;
  table.profileVersion = profile.profileVersion;
  llvm::StringSet<> stageIds;

  for (const LogicalStage &stage : partition.stages) {
    if (stage.id.empty() || !stageIds.insert(stage.id).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage ids must be non-empty and unique: '%s'", stage.id.c_str());
    if (stage.iterationCount <= 0 || !stage.features.isValid() ||
        !stage.workload.isFiniteAndNonNegative())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage '%s' has invalid iteration/features", stage.id.c_str());
    if (!stage.simdLegal && !stage.simtLegal)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Stage '%s' has no legal StageMode",
                                     stage.id.c_str());
    if (stage.simtLegal && stage.legalSimtFactors.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "SIMT Stage '%s' has no legal SuperBlock factor", stage.id.c_str());

    LogicalStageCost logicalCost;
    logicalCost.id = stage.id;
    logicalCost.model = stringifyStageCostModel(stage.costModelKind).str();
    logicalCost.schedule = stage.scheduleKind;
    logicalCost.iterationCount = stage.iterationCount;
    logicalCost.features = stage.features;
    logicalCost.workload = stage.workload;
    logicalCost.ownedOperationCount =
        static_cast<int64_t>(stage.operations.size());
    logicalCost.sourceLocations = collectSourceLocations(stage);
    logicalCost.liveInCount = static_cast<int64_t>(stage.liveIns.size());
    logicalCost.liveOutCount = static_cast<int64_t>(stage.liveOuts.size());
    logicalCost.liveInBytes = stage.liveInBytes;
    logicalCost.liveOutBytes = stage.liveOutBytes;
    logicalCost.localSimtScopeCount = stage.localSimtScopeCount;
    logicalCost.scopeInputTensorBytes = stage.scopeInputTensorBytes;
    logicalCost.scopeOutputTensorBytes = stage.scopeOutputTensorBytes;
    logicalCost.simtAnchorIndices = stage.simtAnchorIndices;
    logicalCost.localSimtMaterializable = stage.localSimtMaterializable;
    logicalCost.localSuperblockMaterializable =
        stage.localSuperblockMaterializable;
    logicalCost.legalSimtFactors = stage.legalSimtFactors;
    logicalCost.localSimtFactors = stage.localSimtFactors;

    llvm::SmallVector<StageImplementation> implementations;
    if (stage.simdLegal)
      implementations.push_back({StageMode::SIMD, 1, false});
    if (stage.simtLegal)
      for (int64_t factor : stage.legalSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, false});
    if (stage.simtLegal && stage.localSimtMaterializable)
      for (int64_t factor : stage.localSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, true});

    for (const StageImplementation &implementation : implementations) {
      if (!isDeclaredLegal(stage, implementation))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has an illegal candidate",
                                       stage.id.c_str());
      auto fittedLoad = calibratedIndirectLoad(stage, profile, implementation);
      auto fittedStore =
          calibratedIndirectStore(stage, profile, implementation);
      // Fits already describe the full W-warp increment; do not divide by W.
      StageResourceCycles resources = mapWorkload(
          stage,
          implementation.mode == StageMode::SIMD ? profile.simd : profile.simt,
          implementation.mode, fittedLoad, fittedStore);
      StageImplementationCost cost;
      cost.implementation = implementation;
      if (fittedLoad)
        cost.indirectLoadPricing = implementation.mode == StageMode::SIMD
                                       ? profile.simdIndirectLoadModel
                                       : profile.simtIndirectLoadModel;
      if (fittedStore)
        cost.indirectStorePricing = implementation.mode == StageMode::SIMD
                                        ? profile.simdIndirectStoreModel
                                        : profile.simtIndirectStoreModel;
      cost.resources = resources;
      cost.totalCycles = applySuperBlock(
          stage, resources, implementation, profile,
          estimateStage(stage, profile, implementation.mode, resources));
      if (!cost.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' produced an invalid cost",
                                       stage.id.c_str());
      logicalCost.implementations.push_back(std::move(cost));
    }

    table.stages.push_back(std::move(logicalCost));
  }
  return table;
}
