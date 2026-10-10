//===- StageCostModel.h - Interface for one Stage cost model ---*- C++ -*-===//
#ifndef ASCENDMODEL_ROUTEMODEL_STAGECOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_STAGECOSTMODEL_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include <optional>

namespace mlir::ascend {

/// Prices one legal Stage implementation. Models consume immutable analysis
/// facts; they neither partition IR nor choose a kernel route.
class StageCostModel {
public:
  virtual ~StageCostModel() = default;
  virtual StageImplementationCost
  cost(const LogicalStage &stage, const HardwareProfile &profile,
       const StageImplementation &implementation) const = 0;

protected:
  enum class SuperBlockPolicy { ReplicatedWork, IndependentRecurrences };

  /// Shared per-iteration resource accounting. Optional fits replace only the
  /// target memory resource; helpers and all other resources remain charged.
  static StageResourceCycles
  mapWorkload(const LogicalStage &stage, const HardwareProfile &hardware,
              StageMode mode, std::optional<double> fittedLoad = std::nullopt,
              std::optional<double> fittedStore = std::nullopt);

  /// Common SuperBlock accounting is applied once after the model's latency.
  /// Each model selects its policy; the base does not dispatch on Stage kind.
  static StageImplementationCost
  finishCost(const LogicalStage &stage, const HardwareProfile &profile,
             const StageImplementation &implementation,
             const StageResourceCycles &resources, double stageCycles,
             SuperBlockPolicy policy = SuperBlockPolicy::ReplicatedWork);

  static double iterations(const LogicalStage &stage);
  static double controlBody(const StageResourceCycles &resources);
  static double serialBody(const StageResourceCycles &resources);
  static bool permitsSimdOverlap(const LogicalStage &stage);
  static double genericLatency(const LogicalStage &stage, StageMode mode,
                               const StageResourceCycles &resources);

private:
  static double applySuperBlock(const LogicalStage &stage,
                                const StageResourceCycles &resources,
                                const StageImplementation &implementation,
                                const HardwareProfile &profile,
                                double stageCycles, SuperBlockPolicy policy);
};

/// Stateless models have process lifetime. Simple kinds sharing a formula
/// reuse one implementation; specialized kinds own independent model classes.
const StageCostModel &getStageCostModel(StageCostModelKind kind);

} // namespace mlir::ascend
#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODEL_H
