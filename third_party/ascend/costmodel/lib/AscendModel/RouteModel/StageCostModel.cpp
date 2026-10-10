//===- StageCostModel.cpp - Shared Stage resource accounting
//---------------===//
#include "AscendModel/RouteModel/StageCostModel.h"
#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::ascend;

double StageCostModel::iterations(const LogicalStage &stage) {
  return static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
}

double StageCostModel::controlBody(const StageResourceCycles &resources) {
  return resources.loopControl + resources.branchControl +
         resources.divergence + resources.synchronization;
}

double StageCostModel::serialBody(const StageResourceCycles &resources) {
  const double execution =
      resources.scalar + resources.load + resources.store + resources.atomic +
      resources.compute + resources.predicate + resources.shuffle +
      resources.dot + controlBody(resources) + resources.spill;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

bool StageCostModel::permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
}

namespace {
// Scalar white-box terms, already in the profile's SYS_CNT cycle domain.
static double mainScalarLoadCycles(double count,
                                   const StageModeProfile &profile) {
  const double k = std::max(1.0, count);
  const double perLine = k <= profile.mainScalarLoadExtraLineHighThreshold
                             ? profile.mainScalarLoadExtraLineLowCycles
                             : profile.mainScalarLoadExtraLineHighCycles;
  return profile.mainScalarLoadPrepCycles + profile.mainScalarLoadFillCycles +
         std::max(0.0, k - profile.mainScalarLoadOutstandingLines) * perLine +
         (k - 1.0) * profile.mainScalarLoadIssueCycles;
}

static double simtUniformLoadCycles(double count,
                                    const StageModeProfile &profile) {
  return profile.simtUniformLoadPrepCycles + profile.simtUniformLoadFillCycles +
         (std::max(1.0, count) - 1.0) *
             profile.simtUniformLoadDiffLineIssueCycles;
}

static double mte3StoreCycles(const StageModeProfile &profile) {
  return profile.mte3StorePrepCycles + profile.mte3StoreFillCycles;
}

static double simtUniformStoreCycles(const StageModeProfile &profile) {
  return profile.simtUniformStoreBaseCycles;
}

static StageResourceCycles
materializeControlFlow(const LogicalStage &stage, StageMode mode,
                       StageResourceCycles resources,
                       const StageControlFlowRates &rates) {
  resources.loopControl +=
      static_cast<double>(stage.features.loopBackedgeCount) *
      rates.loopBackedgeCycles;
  resources.branchControl +=
      static_cast<double>(stage.features.conditionalBranchCount) *
      rates.conditionalBranchCycles;
  resources.synchronization +=
      static_cast<double>(stage.features.synchronizationCount) *
      rates.synchronizationCycles;
  if (mode == StageMode::SIMT) {
    resources.divergence +=
        static_cast<double>(stage.features.divergentBranchCount) *
        (1.0 - stage.features.activeLaneRatio) *
        rates.divergentBranchPenaltyCycles;
  }
  return resources;
}

} // namespace

StageResourceCycles StageCostModel::mapWorkload(
    const LogicalStage &stage, const HardwareProfile &hardware, StageMode mode,
    std::optional<double> fittedLoad, std::optional<double> fittedStore) {
  const StageModeProfile &profile =
      mode == StageMode::SIMD ? hardware.simd : hardware.simt;
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  const bool simd = mode == StageMode::SIMD;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  llvm::StringMap<double> describedElements;
  llvm::StringMap<double> describedVectorInstructions;
  double describedIssueElements = 0.0;
  double describedIssueInstructions = 0.0;
  if (simd) {
    for (const TensorOperationWorkload &tensor :
         work.tensorOperationWorkloads) {
      describedElements[tensor.operation] += tensor.logicalElements;
      describedIssueElements += tensor.logicalElements;
      const double segmentBits =
          static_cast<double>(tensor.contiguousElementsPerSegment) *
          static_cast<double>(tensor.elementBitWidth);
      const double vectorInstructions =
          tensor.segmentCount *
          std::ceil(segmentBits / static_cast<double>(profile.vectorWidthBits));
      describedVectorInstructions[tensor.operation] += vectorInstructions;
      describedIssueInstructions += vectorInstructions;
    }
  }
  for (const auto &[name, elements] : work.operationElements) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    double instructions = elements;
    if (simd) {
      const double described = describedElements.lookup(name);
      const double tolerance = 1e-9 * std::max(1.0, elements);
      instructions =
          std::abs(described - elements) <= tolerance
              ? describedVectorInstructions.lookup(name)
              : std::ceil(elements / static_cast<double>(profile.vectorWidth));
    }
    resources.compute +=
        instructions / rate->second.throughput * rate->second.factor;
  }
  resources.scalar += work.scalarOperations / profile.scalarOperationsPerCycle;
  const double directLoadBytes = work.loadBytes - work.indirectLoadBytes;
  const double directStoreBytes = work.storeBytes - work.indirectStoreBytes;
  const double directLoadInstructions =
      work.loadWarpInstructions - work.indirectLoadTransactions;
  const double directStoreInstructions =
      work.storeWarpInstructions - work.indirectStoreTransactions;
  if (simd) {
    resources.load = directLoadBytes / profile.loadBytesPerCycle;
    resources.store = directStoreBytes / profile.storeBytesPerCycle;
  } else {
    resources.load =
        directLoadInstructions / profile.loadWarpInstructionsPerCycle;
    resources.store =
        directStoreInstructions / profile.storeWarpInstructionsPerCycle;
  }
  resources.load += fittedLoad.value_or(
      work.indirectLoadTransactions / profile.indirectLoadTransactionsPerCycle);
  resources.store +=
      fittedStore.value_or(work.indirectStoreTransactions /
                           profile.indirectStoreTransactionsPerCycle);
  // Preserve one uncovered loaded-index dependency latency per Stage
  // iteration, but charge it only when an actual indirect access exists.
  if (work.indirectLoadTransactions > 0.0) {
    // Matched-load difference already includes the induced index wait.
    if (!fittedLoad)
      resources.load += profile.indirectDependencyLatencyCycles;
  } else if (work.indirectStoreTransactions > 0.0 && !fittedStore)
    // The matched-store increment replaces the legacy write/wait estimate.
    resources.store += profile.indirectDependencyLatencyCycles;

  for (const AtomicWorkload &atomic : work.atomicWorkloads) {
    auto rate = profile.atomicRates.find(atomic.profileKey());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find("default");
    if (rate == profile.atomicRates.end())
      continue;
    const StageAtomicRate &atomicRate = rate->second;
    const double base =
        atomic.logicalOperationInstances * atomicRate.operationStartupCycles +
        atomic.logicalElements / atomicRate.logicalElementsPerCycle;
    const double contention =
        atomic.contentionUnknown ? atomicRate.unknownContentionMultiplier : 1.0;
    resources.atomic += base * contention;
    if (atomic.resultUsed)
      resources.atomic +=
          atomic.logicalOperationInstances * atomicRate.resultDependencyCycles;
  }
  if (work.scalarLoadCount > 0.0) {
    resources.load +=
        simd ? mainScalarLoadCycles(work.scalarLoadCount, profile)
             : simtUniformLoadCycles(work.scalarLoadCount, profile);
  }
  if (work.scalarStoreCount > 0.0) {
    resources.store +=
        simd ? mte3StoreCycles(profile) : simtUniformStoreCycles(profile);
  }
  double predicateInstructions = work.predicateElements;
  if (simd) {
    const double described = describedElements.lookup("predicate.cmp");
    const double tolerance = 1e-9 * std::max(1.0, work.predicateElements);
    predicateInstructions =
        std::abs(described - work.predicateElements) <= tolerance
            ? describedVectorInstructions.lookup("predicate.cmp")
            : std::ceil(work.predicateElements /
                        static_cast<double>(profile.vectorWidth));
  }
  resources.predicate =
      predicateInstructions / profile.predicateOperationsPerCycle;
  resources.shuffle = work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  resources.scanShuffle =
      work.scanShuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (work.dotFlops > 0.0) {
    resources.setup += profile.dotSetupCycles;
    resources.dot = work.dotFlops / profile.dotFlopsPerCycle;
  }
  double issueInstructions =
      std::ceil(work.issueElements / static_cast<double>(profile.issueWidth));
  if (simd) {
    // Keep the issue floor consistent with operation pricing.  Aggregating
    // unrelated short rows before dividing by the vector width makes several
    // independently issued instructions look like one full-width operation.
    const double undescribedIssueElements =
        std::max(0.0, work.issueElements - describedIssueElements);
    issueInstructions = describedIssueInstructions +
                        std::ceil(undescribedIssueElements /
                                  static_cast<double>(profile.issueWidth));
  }
  resources.issue = issueInstructions / profile.issueOperationsPerCycle;
  resources.spill =
      work.estimatedSpillTransactions / profile.spillTransactionsPerCycle;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.shuffle +
                             resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath =
        resources.compute + resources.predicate + resources.shuffle;
  return materializeControlFlow(stage, mode, resources, profile.controlFlow);
}

double StageCostModel::applySuperBlock(
    const LogicalStage &stage, const StageResourceCycles &resources,
    const StageImplementation &implementation, const HardwareProfile &profile,
    double stageCycles, SuperBlockPolicy policy) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;

  const double factor = static_cast<double>(implementation.superblockFactor);
  const double effectiveFactor = std::min(
      factor, static_cast<double>(profile.superblockUsefulFactorLimit));
  const double latencySensitivePerIteration =
      resources.load + resources.store + resources.atomic + resources.shuffle +
      resources.divergence;
  const double latencySensitive =
      iterations(stage) * latencySensitivePerIteration;
  // SuperBlock creates `factor` independent logical-program groups on one
  // physical core.  It can hide latency across those groups, but it cannot
  // divide dependent arithmetic, loop control, or synchronization.
  const double pressure =
      iterations(stage) * resources.spill * std::max(0.0, factor - 1.0);
  // Live-out bytes alone do not prove register pressure: they describe the
  // Stage ABI, not the allocator's simultaneously-live set.  Charge replicated
  // persistent state only when workload analysis has independently predicted
  // spill traffic.  This keeps the penalty evidence based and lets independent
  // recurrence groups use F4 when the generated SIMT VF has no STK/LDK.
  const double persistentStatePressure =
      stage.features.hasLoopCarriedDataDependency && resources.spill > 0.0
          ? std::max(
                0.0,
                factor -
                    static_cast<double>(
                        profile.superblockPersistentStatePressureFreeFactor)) *
                static_cast<double>(stage.liveOutBytes) /
                profile.superblockPersistentStateBytesPerCycle
          : 0.0;
  const double fixed = resources.setup;
  const double issueFloor =
      fixed + factor * iterations(stage) * resources.issue;
  // A recurrence is serial inside one logical program.  SuperBlock contributes
  // F independent logical programs to the same physical program, allowing the
  // scheduler to cover one program's dependency stalls with another program.
  // Normalize the critical-path portion per logical program, but retain the
  // aggregate issue floor: a larger factor cannot create additional issue
  // bandwidth.
  // This applies equally to whole-kernel and scope-local SuperBlock because
  // both materializers batch complete logical programs around the Stage.
  if (policy == SuperBlockPolicy::IndependentRecurrences) {
    const double recurrenceBody = std::max(0.0, stageCycles - fixed);
    return std::max(issueFloor, fixed + recurrenceBody + pressure) +
           persistentStatePressure;
  }
  // Proven persistent-state pressure is additional register/stack work and
  // cannot disappear behind the ordinary issue floor.
  const double body = std::max(0.0, stageCycles - fixed);
  const double groupedBody = factor * std::max(0.0, body - latencySensitive) +
                             factor * latencySensitive / effectiveFactor;
  return std::max(issueFloor, fixed + groupedBody + pressure) +
         persistentStatePressure;
}

double StageCostModel::genericLatency(const LogicalStage &stage, StageMode mode,
                                      const StageResourceCycles &r) {
  if (mode == StageMode::SIMD)
    return r.setup +
           iterations(stage) *
               (std::max({r.load, r.store, r.atomic,
                          r.compute + r.dot + r.shuffle,
                          r.scalar + r.predicate + controlBody(r), r.issue}) +
                r.spill);
  return r.setup + iterations(stage) * serialBody(r);
}

StageImplementationCost
StageCostModel::finishCost(const LogicalStage &stage,
                           const HardwareProfile &profile,
                           const StageImplementation &implementation,
                           const StageResourceCycles &resources,
                           double stageCycles, SuperBlockPolicy policy) {
  StageImplementationCost result;
  result.implementation = implementation;
  result.resources = resources;
  result.totalCycles = applySuperBlock(stage, resources, implementation,
                                       profile, stageCycles, policy);
  return result;
}
