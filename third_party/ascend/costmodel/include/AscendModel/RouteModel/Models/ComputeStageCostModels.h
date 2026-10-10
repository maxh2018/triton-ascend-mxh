//===- ComputeStageCostModels.h - Specialized Stage models ---*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_MODELS_COMPUTESTAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_MODELS_COMPUTESTAGECOSTMODELS_H
#include "AscendModel/RouteModel/StageCostModel.h"

namespace mlir::ascend {

/// Shared resource policy for simple kinds with identical formulas.
class GenericStageCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

/// Overlap independent loop resources only with proven SIMD pipelining.
class IndependentPipelinedLoopCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

/// Cube/Vector overlap policy shared by regular and tiny cube stages.
class CubeRooflineCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

/// Conversion-specific compute/memory overlap policy.
class ConversionPackCostModel final : public StageCostModel {
public:
  StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const override;
};

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_MODELS_COMPUTESTAGECOSTMODELS_H
