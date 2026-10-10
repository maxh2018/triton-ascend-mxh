// Dispatch and recurrence models.
#include "AscendModel/RouteModel/Models/ControlStageCostModels.h"
#include <algorithm>
#include <cmath>

using namespace mlir::ascend;

StageImplementationCost
AutoBlockifyCostModel::cost(const LogicalStage &stage,
                            const HardwareProfile &profile,
                            const StageImplementation &implementation) const {
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

double LoopCarriedRecurrenceCostModel::latency(const LogicalStage &stage,
                                               const HardwareProfile &profile,
                                               StageMode mode,
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
  const int64_t groups =
      std::max<int64_t>(1, std::min(stage.features.parallelRecurrenceGroupCount,
                                    profile.logicalWarpGroupCount));
  return r.setup +
         std::max(std::ceil(count / static_cast<double>(groups)) * critical,
                  count * r.issue);
}

StageImplementationCost LoopCarriedRecurrenceCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
  const auto mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  return finishCost(stage, profile, implementation, r,
                    latency(stage, profile, mode, r),
                    SuperBlockPolicy::IndependentRecurrences);
}
