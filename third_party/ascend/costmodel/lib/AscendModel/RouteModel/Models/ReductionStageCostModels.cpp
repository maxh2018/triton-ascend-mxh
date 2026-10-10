// Reduction and scan models.
#include "AscendModel/RouteModel/Models/ReductionStageCostModels.h"
#include <algorithm>

using namespace mlir::ascend;

StageImplementationCost RowwiseReductionCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
  const auto mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  return finishCost(stage, profile, implementation, r,
                    r.setup + iterations(stage) *
                                  std::max(r.scalar + r.load + r.store +
                                               r.atomic + r.criticalPath +
                                               controlBody(r) + r.spill,
                                           r.issue));
}

StageImplementationCost
PrefixScanCostModel::cost(const LogicalStage &stage,
                          const HardwareProfile &profile,
                          const StageImplementation &implementation) const {
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
