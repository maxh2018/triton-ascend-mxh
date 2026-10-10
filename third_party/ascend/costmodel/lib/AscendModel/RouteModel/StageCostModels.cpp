//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"
#include "AscendModel/RouteModel/StageCostModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {
static std::vector<std::string>
collectSourceLocations(const LogicalStage &stage) {
  std::vector<std::string> result;
  llvm::StringSet<> seen;
  for (Operation *operation : stage.operations) {
    std::string location;
    llvm::raw_string_ostream stream(location);
    operation->getLoc().print(stream);
    stream.flush();
    if (location.empty() || !seen.insert(location).second)
      continue;
    result.push_back(std::move(location));
  }
  return result;
}

static bool isDeclaredLegal(const LogicalStage &stage,
                            const StageImplementation &implementation) {
  if (!implementation.isValid())
    return false;
  if (implementation.mode == StageMode::SIMD)
    return stage.simdLegal && implementation.superblockFactor == 1 &&
           !implementation.localScope;
  if (!stage.simtLegal)
    return false;
  if (implementation.localScope)
    return stage.localSimtMaterializable &&
           llvm::is_contained(stage.localSimtFactors,
                              implementation.superblockFactor);
  return llvm::is_contained(stage.legalSimtFactors,
                            implementation.superblockFactor);
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageCostModel(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
    return "auto_blockify_dispatch";
  case StageCostModelKind::AutoBlockifyLoop:
    return "auto_blockify_loop";
  case StageCostModelKind::ScalarIssue:
    return "scalar_issue";
  case StageCostModelKind::ScalarControl:
    return "scalar_control";
  case StageCostModelKind::ScalarMath:
    return "scalar_math";
  case StageCostModelKind::ScalarLoad:
    return "scalar_load";
  case StageCostModelKind::ScalarStore:
    return "scalar_store";
  case StageCostModelKind::IndexGeneration:
    return "index_generation";
  case StageCostModelKind::PredicateMask:
    return "predicate_mask";
  case StageCostModelKind::LoopPredicate:
    return "loop_predicate";
  case StageCostModelKind::ContinuousTileMemory:
    return "continuous_tile_memory";
  case StageCostModelKind::PartialContinuousTileMemory:
    return "partial_continuous_tile_memory";
  case StageCostModelKind::ContinuousTileStore:
    return "continuous_tile_store";
  case StageCostModelKind::ContinuousShortLoad:
    return "continuous_short_load";
  case StageCostModelKind::CachePolicyStore:
    return "cache_policy_store";
  case StageCostModelKind::IndirectScalarMemory:
    return "indirect_scalar_memory";
  case StageCostModelKind::IndirectGatherMemory:
    return "indirect_gather_memory";
  case StageCostModelKind::AtomicMemory:
    return "atomic_memory";
  case StageCostModelKind::IndependentPipelinedLoop:
    return "independent_pipelined_loop";
  case StageCostModelKind::LoopCarriedRecurrence:
    return "loop_carried_recurrence";
  case StageCostModelKind::RowwiseReduction:
    return "rowwise_reduction";
  case StageCostModelKind::PrefixScan:
    return "prefix_scan";
  case StageCostModelKind::CubeRoofline:
    return "cube_roofline";
  case StageCostModelKind::TinyCubeRoofline:
    return "tiny_cube_roofline";
  case StageCostModelKind::ConversionPack:
    return "conversion_pack";
  }
  llvm_unreachable("unknown StageCostModelKind");
}

bool StageControlFlowRates::isFiniteAndNonNegative() const {
  const std::array<double, 4> values = {
      loopBackedgeCycles, conditionalBranchCycles, divergentBranchPenaltyCycles,
      synchronizationCycles};
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

bool StageAtomicRate::isValid() const {
  return std::isfinite(logicalElementsPerCycle) &&
         logicalElementsPerCycle > 0.0 &&
         std::isfinite(operationStartupCycles) &&
         operationStartupCycles >= 0.0 &&
         std::isfinite(resultDependencyCycles) &&
         resultDependencyCycles >= 0.0 &&
         std::isfinite(unknownContentionMultiplier) &&
         unknownContentionMultiplier >= 1.0;
}

bool StageModeProfile::isValid(StageMode mode) const {
  const std::array<double, 14> common = {setupCycles,
                                         predicateOperationsPerCycle,
                                         shuffleLanesPerCycle,
                                         dotSetupCycles,
                                         dotFlopsPerCycle,
                                         scalarOperationsPerCycle,
                                         issueOperationsPerCycle,
                                         spillTransactionsPerCycle,
                                         indirectLoadTransactionsPerCycle,
                                         indirectStoreTransactionsPerCycle,
                                         prefixScanDependencyFactor,
                                         static_cast<double>(vectorWidthBits),
                                         static_cast<double>(vectorWidth),
                                         static_cast<double>(issueWidth)};
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(indirectDependencyLatencyCycles) ||
      indirectDependencyLatencyCycles < 0.0 ||
      !controlFlow.isFiniteAndNonNegative())
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  return llvm::all_of(operationRates,
                      [](const auto &entry) {
                        return std::isfinite(entry.second.throughput) &&
                               entry.second.throughput > 0.0 &&
                               std::isfinite(entry.second.factor) &&
                               entry.second.factor > 0.0;
                      }) &&
         atomicRates.contains("default") &&
         llvm::all_of(atomicRates,
                      [](const auto &entry) { return entry.second.isValid(); });
}

bool HardwareProfile::isValid() const {
  return !profileVersion.empty() && !target.empty() &&
         (simtIndirectLoadModel.empty() ||
          ((simtIndirectLoadModel == "random_i32_six_term_20261007" ||
            simtIndirectLoadModel == "random_dtype_six_term_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         (simdIndirectLoadModel.empty() ||
          ((simdIndirectLoadModel == "random_f32_matched_ab_20261007" ||
            simdIndirectLoadModel == "random_dtype_matched_ab_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         (simtIndirectStoreModel.empty() ||
          ((simtIndirectStoreModel == "random_f32_store_fill_ab_20261007" ||
            simtIndirectStoreModel == "random_store_no_fill_first_20261008" ||
            simtIndirectStoreModel == "random_store_no_fill_reuse_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         (simdIndirectStoreModel.empty() ||
          ((simdIndirectStoreModel ==
                "random_f32_store_no_fill_first_20261007" ||
            simdIndirectStoreModel ==
                "random_f32_store_no_fill_reuse_20261007" ||
            simdIndirectStoreModel == "random_store_no_fill_first_20261008" ||
            simdIndirectStoreModel == "random_store_no_fill_reuse_20261008") &&
           target == "Ascend950PR/dav-c310")) &&
         logicalWarpGroupCount > 0 && superblockUsefulFactorLimit > 0 &&
         superblockPersistentStatePressureFreeFactor > 0 &&
         superblockPersistentStatePressureFreeFactor <=
             superblockUsefulFactorLimit &&
         std::isfinite(superblockPersistentStateBytesPerCycle) &&
         superblockPersistentStateBytesPerCycle > 0.0 &&
         simd.isValid(StageMode::SIMD) && simt.isValid(StageMode::SIMT) &&
         transition.isValid();
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StagePartition &partition,
                             const HardwareProfile &profile) const {
  if (partition.stages.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires at least one Stage");
  if (!profile.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "HardwareProfile is invalid");
  StageCostTable table;
  table.operationOwnershipComplete = partition.operationOwnershipComplete;
  table.modeledOperationCount = partition.modeledOperationCount;
  table.profileVersion = profile.profileVersion;
  llvm::StringSet<> stageIds;

  for (const LogicalStage &stage : partition.stages) {
    if (stage.id.empty() || !stageIds.insert(stage.id).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage ids must be non-empty and unique: '%s'", stage.id.c_str());
    if (stage.iterationCount <= 0 || !stage.features.isValid() ||
        !stage.workload.isFiniteAndNonNegative())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage '%s' has invalid iteration/features", stage.id.c_str());
    if (!stage.simdLegal && !stage.simtLegal)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Stage '%s' has no legal StageMode",
                                     stage.id.c_str());
    if (stage.simtLegal && stage.legalSimtFactors.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "SIMT Stage '%s' has no legal SuperBlock factor", stage.id.c_str());

    LogicalStageCost logicalCost;
    logicalCost.id = stage.id;
    logicalCost.model = stringifyStageCostModel(stage.costModelKind).str();
    logicalCost.schedule = stage.scheduleKind;
    logicalCost.iterationCount = stage.iterationCount;
    logicalCost.features = stage.features;
    logicalCost.workload = stage.workload;
    logicalCost.ownedOperationCount =
        static_cast<int64_t>(stage.operations.size());
    logicalCost.sourceLocations = collectSourceLocations(stage);
    logicalCost.liveInCount = static_cast<int64_t>(stage.liveIns.size());
    logicalCost.liveOutCount = static_cast<int64_t>(stage.liveOuts.size());
    logicalCost.liveInBytes = stage.liveInBytes;
    logicalCost.liveOutBytes = stage.liveOutBytes;
    logicalCost.localSimtScopeCount = stage.localSimtScopeCount;
    logicalCost.scopeInputTensorBytes = stage.scopeInputTensorBytes;
    logicalCost.scopeOutputTensorBytes = stage.scopeOutputTensorBytes;
    logicalCost.simtAnchorIndices = stage.simtAnchorIndices;
    logicalCost.localSimtMaterializable = stage.localSimtMaterializable;
    logicalCost.localSuperblockMaterializable =
        stage.localSuperblockMaterializable;
    logicalCost.legalSimtFactors = stage.legalSimtFactors;
    logicalCost.localSimtFactors = stage.localSimtFactors;

    llvm::SmallVector<StageImplementation> implementations;
    if (stage.simdLegal)
      implementations.push_back({StageMode::SIMD, 1, false});
    if (stage.simtLegal)
      for (int64_t factor : stage.legalSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, false});
    if (stage.simtLegal && stage.localSimtMaterializable)
      for (int64_t factor : stage.localSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, true});

    const StageCostModel &model = getStageCostModel(stage.costModelKind);
    for (const StageImplementation &implementation : implementations) {
      if (!isDeclaredLegal(stage, implementation))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has an illegal candidate",
                                       stage.id.c_str());
      StageImplementationCost cost = model.cost(stage, profile, implementation);
      if (!cost.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' produced an invalid cost",
                                       stage.id.c_str());
      logicalCost.implementations.push_back(std::move(cost));
    }

    table.stages.push_back(std::move(logicalCost));
  }
  return table;
}
