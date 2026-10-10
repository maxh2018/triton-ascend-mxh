// Compute and loop overlap models.
#include "AscendModel/RouteModel/Models/ComputeStageCostModels.h"
#include <algorithm>
#include <cmath>

using namespace mlir::ascend;

StageImplementationCost
GenericStageCostModel::cost(const LogicalStage &stage,
                            const HardwareProfile &profile,
                            const StageImplementation &implementation) const {
  const auto mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  return finishCost(stage, profile, implementation, r,
                    genericLatency(stage, mode, r));
}

StageImplementationCost IndependentPipelinedLoopCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
  const StageMode mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  double cycles = serial;
  if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
    cycles =
        r.setup +
        count *
            (std::max({r.load, r.store, r.atomic, r.compute + r.dot + r.shuffle,
                       r.scalar + r.predicate + controlBody(r), r.issue}) +
             r.spill);
  return finishCost(stage, profile, implementation, r, cycles);
}

StageImplementationCost
CubeRooflineCostModel::cost(const LogicalStage &stage,
                            const HardwareProfile &profile,
                            const StageImplementation &implementation) const {
  const StageMode mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  double cycles = serial;
  if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
    cycles =
        r.setup +
        count *
            (r.scalar + r.predicate + controlBody(r) + r.shuffle + r.spill +
             std::max({r.load, r.compute + r.dot, r.store, r.atomic, r.issue}));
  return finishCost(stage, profile, implementation, r, cycles);
}

StageImplementationCost
ConversionPackCostModel::cost(const LogicalStage &stage,
                              const HardwareProfile &profile,
                              const StageImplementation &implementation) const {
  const StageMode mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  double cycles = serial;
  if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
    cycles = r.setup + count * (r.predicate + controlBody(r) + r.spill +
                                std::max({r.scalar + r.compute, r.load, r.store,
                                          r.atomic, r.issue}));
  return finishCost(stage, profile, implementation, r, cycles);
}
