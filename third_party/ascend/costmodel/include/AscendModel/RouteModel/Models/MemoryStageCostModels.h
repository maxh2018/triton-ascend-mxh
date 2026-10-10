//===- MemoryStageCostModels.h - Specialized Stage models ---*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_MODELS_MEMORYSTAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_MODELS_MEMORYSTAGECOSTMODELS_H
#include "AscendModel/RouteModel/StageCostModel.h"

namespace mlir::ascend {

/// Direct memory overlap policy, including separately priced atomics.
class ContinuousMemoryCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

/// Unstructured prefixes with a structured suffix retain serial pricing.
class PartialContinuousMemoryCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_MODELS_MEMORYSTAGECOSTMODELS_H
