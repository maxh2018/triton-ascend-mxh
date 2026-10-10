// Simple Stage models share one translation unit; complex indirect memory is
// implemented separately. Each model still owns its cost formula.
#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "AscendModel/RouteModel/StageCostModel.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cmath>

using namespace mlir::ascend;

namespace {
class GenericStageCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const auto mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    return finishCost(stage, profile, implementation, r,
                      genericLatency(stage, mode, r));
  }
};

class IndependentPipelinedLoopCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const StageMode mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    const double count = iterations(stage);
    const double serial = r.setup + count * serialBody(r);
    double cycles = serial;
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      cycles =
          r.setup + count * (std::max({r.load, r.store, r.atomic,
                                       r.compute + r.dot + r.shuffle,
                                       r.scalar + r.predicate + controlBody(r),
                                       r.issue}) +
                             r.spill);
    return finishCost(stage, profile, implementation, r, cycles);
  }
};

class CubeRooflineCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const StageMode mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    const double count = iterations(stage);
    const double serial = r.setup + count * serialBody(r);
    double cycles = serial;
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      cycles = r.setup + count * (r.scalar + r.predicate + controlBody(r) +
                                  r.shuffle + r.spill +
                                  std::max({r.load, r.compute + r.dot, r.store,
                                            r.atomic, r.issue}));
    return finishCost(stage, profile, implementation, r, cycles);
  }
};

class ConversionPackCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const StageMode mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    const double count = iterations(stage);
    const double serial = r.setup + count * serialBody(r);
    double cycles = serial;
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      cycles = r.setup + count * (r.predicate + controlBody(r) + r.spill +
                                  std::max({r.scalar + r.compute, r.load,
                                            r.store, r.atomic, r.issue}));
    return finishCost(stage, profile, implementation, r, cycles);
  }
};

class AutoBlockifyCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const auto mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    const double count =
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop
            ? iterations(stage)
            : 1.0;
    return finishCost(stage, profile, implementation, r,
                      r.setup +
                          count * std::max(r.scalar + controlBody(r), r.issue));
  }
};

class LoopCarriedRecurrenceCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const auto mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    return finishCost(stage, profile, implementation, r,
                      latency(stage, profile, mode, r),
                      SuperBlockPolicy::IndependentRecurrences);
  }

private:
  static double latency(const LogicalStage &stage,
                        const HardwareProfile &profile, StageMode mode,
                        const StageResourceCycles &r) {
    const double count = iterations(stage);
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
};

class ContinuousMemoryCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const StageMode mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    const double count = iterations(stage);
    const double serial = r.setup + count * serialBody(r);
    double cycles = serial;
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      cycles =
          r.setup + count * (r.scalar + r.predicate + controlBody(r) + r.spill +
                             std::max({r.load, r.store, r.atomic, r.issue}));
    return finishCost(stage, profile, implementation, r, cycles);
  }
};

class PartialContinuousMemoryCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const auto mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    // Unstructured slices are serial, even when the Stage permits SIMD overlap.
    return finishCost(stage, profile, implementation, r,
                      r.setup + iterations(stage) * serialBody(r));
  }
};

class RowwiseReductionCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const auto mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    return finishCost(stage, profile, implementation, r,
                      r.setup + iterations(stage) *
                                    std::max(r.scalar + r.load + r.store +
                                                 r.atomic + r.criticalPath +
                                                 controlBody(r) + r.spill,
                                             r.issue));
  }
};

class PrefixScanCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override {
    const auto mode = implementation.mode;
    const auto r = mapWorkload(stage, profile, mode);
    const auto &modeProfile =
        mode == StageMode::SIMD ? profile.simd : profile.simt;
    const double scanCritical =
        r.compute + r.predicate +
        r.shuffle * modeProfile.prefixScanDependencyFactor;
    return finishCost(stage, profile, implementation, r,
                      r.setup + iterations(stage) *
                                    std::max(r.scalar + r.load + r.store +
                                                 r.atomic + scanCritical +
                                                 controlBody(r) + r.spill,
                                             r.issue));
  }
};

template <typename Model> const StageCostModel &model() {
  static const Model instance;
  return instance;
}
} // namespace

const StageCostModel &mlir::ascend::getStageCostModel(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop:
    return model<AutoBlockifyCostModel>();
  case StageCostModelKind::PartialContinuousTileMemory:
    return model<PartialContinuousMemoryCostModel>();
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
  case StageCostModelKind::AtomicMemory:
    return model<ContinuousMemoryCostModel>();
  case StageCostModelKind::IndirectGatherMemory:
    return model<IndirectGatherMemoryCostModel>();
  case StageCostModelKind::IndependentPipelinedLoop:
    return model<IndependentPipelinedLoopCostModel>();
  case StageCostModelKind::LoopCarriedRecurrence:
    return model<LoopCarriedRecurrenceCostModel>();
  case StageCostModelKind::RowwiseReduction:
    return model<RowwiseReductionCostModel>();
  case StageCostModelKind::PrefixScan:
    return model<PrefixScanCostModel>();
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    return model<CubeRooflineCostModel>();
  case StageCostModelKind::ConversionPack:
    return model<ConversionPackCostModel>();
  case StageCostModelKind::ScalarIssue:
  case StageCostModelKind::ScalarControl:
  case StageCostModelKind::ScalarMath:
  case StageCostModelKind::ScalarLoad:
  case StageCostModelKind::ScalarStore:
  case StageCostModelKind::IndexGeneration:
  case StageCostModelKind::PredicateMask:
  case StageCostModelKind::LoopPredicate:
  case StageCostModelKind::IndirectScalarMemory:
    return model<GenericStageCostModel>();
  }
  llvm_unreachable("unknown StageCostModelKind");
}
