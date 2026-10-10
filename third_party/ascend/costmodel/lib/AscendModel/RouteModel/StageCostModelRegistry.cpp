// Kind-to-model dispatch. No formulas or mutable analysis state live here.
#include "AscendModel/RouteModel/Models/ComputeStageCostModels.h"
#include "AscendModel/RouteModel/Models/ControlStageCostModels.h"
#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "AscendModel/RouteModel/Models/MemoryStageCostModels.h"
#include "AscendModel/RouteModel/Models/ReductionStageCostModels.h"
#include "AscendModel/RouteModel/StageCostModel.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::ascend;

namespace {
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
