// Continuous and partially continuous memory.
#include "AscendModel/RouteModel/Models/MemoryStageCostModels.h"
#include <algorithm>

using namespace mlir::ascend;

StageImplementationCost ContinuousMemoryCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
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

StageImplementationCost PartialContinuousMemoryCostModel::cost(
    const LogicalStage &stage, const HardwareProfile &profile,
    const StageImplementation &implementation) const {
  const auto mode = implementation.mode;
  const auto r = mapWorkload(stage, profile, mode);
  // Unstructured slices are serial, even when the Stage permits SIMD overlap.
  return finishCost(stage, profile, implementation, r,
                    r.setup + iterations(stage) * serialBody(r));
}
