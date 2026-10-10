//===- ReductionStageCostModels.h - Specialized Stage models ---*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_MODELS_REDUCTIONSTAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_MODELS_REDUCTIONSTAGECOSTMODELS_H
#include "AscendModel/RouteModel/StageCostModel.h"

namespace mlir::ascend {

/// Reduction critical path and ordinary memory/control resource costs.
class RowwiseReductionCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

/// Scan critical path with the mode-specific shuffle dependency factor.
class PrefixScanCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_MODELS_REDUCTIONSTAGECOSTMODELS_H
