//===- ControlStageCostModels.h - Specialized Stage models ---*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_MODELS_CONTROLSTAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_MODELS_CONTROLSTAGECOSTMODELS_H
#include "AscendModel/RouteModel/StageCostModel.h"

namespace mlir::ascend {

/// Scalar dispatch is paid once; the loop-control body is paid per iteration.
class AutoBlockifyCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

/// Owns recurrence critical paths, persistent state and batching policy.
class LoopCarriedRecurrenceCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;

private:
  static double latency(const LogicalStage &stage,
                        const HardwareProfile &profile, StageMode mode,
                        const StageResourceCycles &resources);
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_MODELS_CONTROLSTAGECOSTMODELS_H
