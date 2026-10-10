// Tests for StageCostModel responsibilities.
#include "AscendModel/RouteModel/StageCostModel.h"
#include "CostModelTestUtils.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(StageCostModelTest, RegistryDispatchMatchesDirectModelCosts) {
  auto profile = hardwareProfile();
  profile.logicalWarpGroupCount = 4;
  profile.simd.prefixScanDependencyFactor = 2.5;
  // StageCostModelKind is a contiguous enum; exercise every registered kind.
  for (int value = int(StageCostModelKind::AutoBlockifyDispatch);
       value <= int(StageCostModelKind::ConversionPack); ++value) {
    const auto kind = static_cast<StageCostModelKind>(value);
    const auto &model = getStageCostModel(kind);
    SCOPED_TRACE(stringifyStageCostModel(kind).str());
    for (bool dependent : {false, true}) {
      auto stage = logicalStage("direct-model", kind,
                                StageScheduleKind::IndependentPipelined, 7);
      stage.legalSimtFactors = {1, 2, 4};
      stage.features.hasLoop = true;
      stage.features.hasLoopCarriedDataDependency = dependent;
      stage.features.hasReduction = true;
      stage.features.hasPrefixScan = true;
      stage.features.parallelRecurrenceGroupCount = 3;
      stage.features.loopBackedgeCount = 2;
      stage.features.conditionalBranchCount = 1;
      stage.features.divergentBranchCount = 1;
      stage.features.synchronizationCount = 1;
      stage.features.activeLaneRatio = 0.75;
      stage.liveOutBytes = 128;
      auto &work = stage.workload;
      work.scalarOperations = 7;
      work.loadBytes = 640;
      work.storeBytes = 320;
      work.loadWarpInstructions = 20;
      work.storeWarpInstructions = 10;
      work.predicateElements = 64;
      work.shuffleLaneSteps = 320;
      work.scanShuffleLaneSteps = 160;
      work.dotFlops = 128;
      work.estimatedSpillTransactions = 2;
      auto table = evaluateOneStage(stage, profile);
      ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
      const auto &costs = table->stages.front().implementations;
      ASSERT_EQ(costs.size(), 4u);
      for (const auto &evaluated : costs) {
        const auto expected =
            model.cost(stage, profile, evaluated.implementation);
        EXPECT_TRUE(expected.isValid());
        const auto json = [](const StageImplementationCost &cost) {
          return llvm::formatv("{0}", llvm::json::Value(cost.toJSON())).str();
        };
        // Includes all resource fields, pricing labels and SuperBlock cost.
        EXPECT_EQ(json(evaluated), json(expected));
      }
    }
  }
}

TEST(StageCostModelTest, DirectModelsPreserveSerialAndPipelinedMemoryPolicies) {
  auto stage = logicalStage("memory", StageCostModelKind::ContinuousTileMemory,
                            StageScheduleKind::IndependentPipelined, 3);
  stage.workload.scalarOperations = 7;
  stage.workload.loadBytes = 640;
  stage.workload.storeBytes = 320;
  auto profile = hardwareProfile();
  const StageImplementation simd{StageMode::SIMD, 1, false};
  auto pipelined = getStageCostModel(StageCostModelKind::ContinuousTileMemory)
                       .cost(stage, profile, simd);
  // setup + iterations * (scalar + max(load, store, atomic, issue)).
  EXPECT_DOUBLE_EQ(pipelined.totalCycles, 10 + 3 * (7 + 20));
  stage.costModelKind = StageCostModelKind::PartialContinuousTileMemory;
  auto serial =
      getStageCostModel(StageCostModelKind::PartialContinuousTileMemory)
          .cost(stage, profile, simd);
  EXPECT_DOUBLE_EQ(serial.totalCycles, 10 + 3 * (7 + 20 + 20 + 1));
  EXPECT_DOUBLE_EQ(serial.resources.load, pipelined.resources.load);
  EXPECT_DOUBLE_EQ(serial.resources.store, pipelined.resources.store);
  stage.costModelKind = StageCostModelKind::ContinuousTileMemory;
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  EXPECT_DOUBLE_EQ(getStageCostModel(StageCostModelKind::ContinuousTileMemory)
                       .cost(stage, profile, simd)
                       .totalCycles,
                   serial.totalCycles);
}

TEST(StageCostModelTest, SimdPricesShortAxesPerSegmentAndElementWidth) {
  auto simdCost = [](int64_t elementBits, int64_t contiguousElements,
                     double segmentCount) {
    LogicalStage stage =
        logicalStage("short-axis", StageCostModelKind::ScalarMath);
    stage.workload.operationElements["f32.add"] =
        static_cast<double>(contiguousElements) * segmentCount;
    TensorOperationWorkload tensor;
    tensor.operation = "f32.add";
    tensor.elementBitWidth = elementBits;
    tensor.logicalElements =
        static_cast<double>(contiguousElements) * segmentCount;
    tensor.segmentCount = segmentCount;
    tensor.contiguousElementsPerSegment = contiguousElements;
    stage.workload.tensorOperationWorkloads.push_back(std::move(tensor));
    auto table = evaluateOneStage(std::move(stage));
    if (!table) {
      ADD_FAILURE() << llvm::toString(table.takeError());
      return mlir::ascend::StageResourceCycles{};
    }
    return table->stages.front().implementations.front().resources;
  };

  // These are already projected segments (e.g. broadcast-constrained rows),
  // not initial TTIR shapes. A short segment still uses a vector instruction.
  auto fourBySixteen = simdCost(32, 16, 4.0);
  auto eightByEight = simdCost(32, 8, 8.0);
  auto sixteenByFour = simdCost(32, 4, 16.0);
  EXPECT_DOUBLE_EQ(fourBySixteen.compute, 4.0);
  EXPECT_DOUBLE_EQ(eightByEight.compute, 8.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.compute, 16.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.scalar, 0.0);
  EXPECT_DOUBLE_EQ(fourBySixteen.issue, 1.0);
  EXPECT_DOUBLE_EQ(eightByEight.issue, 2.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.issue, 4.0);

  // FP16 uses twice as many elements per 256-byte instruction; width is not
  // silently inherited from the old FP32-only vectorWidth field.
  auto fp16Rows = simdCost(16, 128, 2.0);
  EXPECT_DOUBLE_EQ(fp16Rows.compute, 2.0);
}

TEST(StageCostModelTest, PrefixScanUsesModeSpecificDependencyFactor) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationElements.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 64.0;
  stage.workload.shuffleLaneSteps = 320.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.prefixScanDependencyFactor = 2.5;
  profile.simt.prefixScanDependencyFactor = 1.0;
  auto table = evaluateOneStage(std::move(stage), profile);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());

  const auto &implementations = table->stages.front().implementations;
  ASSERT_EQ(implementations.size(), 2u);
  EXPECT_EQ(implementations[0].implementation.mode, StageMode::SIMD);
  EXPECT_EQ(implementations[1].implementation.mode, StageMode::SIMT);
  EXPECT_GT(implementations[0].totalCycles, implementations[1].totalCycles);
}

TEST(StageCostModelTest, LoopCarriedRecurrenceAppliesScanDependencyFactor) {
  auto buildStage = [] {
    LogicalStage stage = logicalStage(
        "recurrence_scan", StageCostModelKind::LoopCarriedRecurrence,
        StageScheduleKind::LoopCarriedSerial, /*iterations=*/4);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.features.hasPrefixScan = true;
    stage.workload.operationElements.clear();
    stage.workload.scalarOperations = 0.0;
    stage.workload.issueElements = 64.0;
    // Mixed shuffle pool: 320 scan-class steps (tt.scan) + 320 reduce-class
    // steps (tt.reduce).
    stage.workload.shuffleLaneSteps = 640.0;
    stage.workload.scanShuffleLaneSteps = 320.0;
    return stage;
  };

  // Identity factors keep the legacy recurrence scoring.
  HardwareProfile baselineProfile = hardwareProfile();
  baselineProfile.simd.prefixScanDependencyFactor = 1.0;
  baselineProfile.simt.prefixScanDependencyFactor = 1.0;
  auto baseline = evaluateOneStage(buildStage(), baselineProfile);
  ASSERT_TRUE(bool(baseline)) << llvm::toString(baseline.takeError());
  ASSERT_EQ(baseline->stages.front().implementations.size(), 2u);

  // Scan factors: SIMD 2.5 / SIMT 1.0 (production profile shape).
  HardwareProfile scanProfile = baselineProfile;
  scanProfile.simd.prefixScanDependencyFactor = 2.5;
  auto scaled = evaluateOneStage(buildStage(), scanProfile);
  ASSERT_TRUE(bool(scaled)) << llvm::toString(scaled.takeError());
  ASSERT_EQ(scaled->stages.front().implementations.size(), 2u);

  // Only the scan-class half is scaled: SIMD scan-shuffle cycles grow from
  // 320/32 = 10 to 10 * 2.5 = 25, so the stage total grows by 4 iterations *
  // (25 - 10) = 60 cycles.  The reduce-class half keeps the ideal rate and
  // SIMT keeps the identity factor, so both must not move.
  EXPECT_DOUBLE_EQ(scaled->stages.front().implementations[0].totalCycles,
                   baseline->stages.front().implementations[0].totalCycles +
                       60.0);
  EXPECT_DOUBLE_EQ(scaled->stages.front().implementations[1].totalCycles,
                   baseline->stages.front().implementations[1].totalCycles);
  EXPECT_GT(scaled->stages.front().implementations[0].totalCycles,
            scaled->stages.front().implementations[1].totalCycles);

  // A recurrence whose shuffle pool is purely reduce-class must not consume
  // the scan factor even though the scan feature flag is set.
  LogicalStage reduceOnlyStage = buildStage();
  reduceOnlyStage.workload.scanShuffleLaneSteps = 0.0;
  auto reduceOnlyScaled =
      evaluateOneStage(std::move(reduceOnlyStage), scanProfile);
  ASSERT_TRUE(bool(reduceOnlyScaled))
      << llvm::toString(reduceOnlyScaled.takeError());
  EXPECT_DOUBLE_EQ(
      reduceOnlyScaled->stages.front().implementations[0].totalCycles,
      baseline->stages.front().implementations[0].totalCycles);

  // A recurrence without a prefix scan must not consume the scan factor.
  LogicalStage plainStage = buildStage();
  plainStage.features.hasPrefixScan = false;
  auto plainScaled = evaluateOneStage(std::move(plainStage), scanProfile);
  ASSERT_TRUE(bool(plainScaled)) << llvm::toString(plainScaled.takeError());
  EXPECT_DOUBLE_EQ(plainScaled->stages.front().implementations[0].totalCycles,
                   baseline->stages.front().implementations[0].totalCycles);
}

TEST(StageCostModelTest, IndependentLoopUsesSimdRooflineAndSerialSimtCost) {
  LogicalStage stage =
      logicalStage("independent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasPointerInduction = true;

  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadWarpInstructions = 20.0;
  stage.workload.storeWarpInstructions = 10.0;
  stage.workload.dotFlops = 512.0;
  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  EXPECT_TRUE(stage.features.permitsSimdRoofline());
  EXPECT_LT(table->stages[0].implementations[0].totalCycles,
            table->stages[0].implementations[1].totalCycles);
}

TEST(StageCostModelTest, SimdCubeStageUsesCvPipelineCriticalPath) {
  LogicalStage stage =
      logicalStage("cube", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasDot = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.operationElements.clear();
  stage.workload.scalarOperations = 16.0;
  stage.workload.loadBytes = 2048.0;
  stage.workload.storeBytes = 512.0;
  stage.workload.dotFlops = 8192.0;
  stage.workload.issueElements = 128.0;

  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());

  const StageImplementationCost &simd = table->stages[0].implementations[0];
  ASSERT_EQ(simd.implementation.mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(simd.resources.setup, 18.0);
  EXPECT_DOUBLE_EQ(simd.resources.load, 64.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.dot, 128.0);
  // SIMD/Cube CV resources overlap inside one Stage.  The model charges the
  // critical resource (MMAD here), rather than serializing load+MMAD+store.
  EXPECT_DOUBLE_EQ(simd.totalCycles, 18.0 + 4.0 * (16.0 + 128.0));
}

TEST(StageCostModelTest, TrueLoopCarriedDependencyDisablesSimdRoofline) {
  LogicalStage stage =
      logicalStage("dependent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.dotFlops = 512.0;

  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  EXPECT_FALSE(stage.features.permitsSimdRoofline());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 0.0);
}

TEST(StageCostModelTest, ControlFlowUsesCountsRatesAndLaneActivity) {
  LogicalStage stage =
      logicalStage("control", StageCostModelKind::ScalarControl,
                   StageScheduleKind::StraightLine, 2);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.conditionalBranchCount = 3;
  stage.features.divergentBranchCount = 2;
  stage.features.loopBackedgeCount = 1;
  stage.features.synchronizationCount = 1;
  stage.features.activeLaneRatio = 0.5;

  stage.workload.scalarOperations = 1.0;

  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  const auto &cost = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(cost.resources.loopControl, 2.0);
  EXPECT_DOUBLE_EQ(cost.resources.branchControl, 9.0);
  EXPECT_DOUBLE_EQ(cost.resources.divergence, 10.0);
  EXPECT_DOUBLE_EQ(cost.resources.synchronization, 7.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 196.0);
}

TEST(StageCostModelTest, RecurrenceAccumulatesCriticalPathAndTraffic) {
  LogicalStage stage =
      logicalStage("recurrence", StageCostModelKind::LoopCarriedRecurrence,
                   StageScheduleKind::LoopCarriedSerial, 4);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.workload.loadWarpInstructions = 10.0;
  stage.workload.storeWarpInstructions = 5.0;
  stage.workload.estimatedSpillTransactions = 7.0;

  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 100.0);
  EXPECT_GT(table->stages[0].implementations[0].resources.criticalPath, 0.0);
}

TEST(StageCostModelTest, SimdRecurrenceChargesPersistentLiveStateBytes) {
  LogicalStage baseline = logicalStage(
      "baseline_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial);
  baseline.features.hasLoop = true;
  baseline.features.hasLoopCarriedDataDependency = true;
  LogicalStage withState = baseline;
  withState.id = "stateful_recurrence";
  withState.liveOutBytes = 800;

  auto baselineTable = evaluateOneStage(baseline);
  auto stateTable = evaluateOneStage(withState);
  ASSERT_TRUE(bool(baselineTable)) << llvm::toString(baselineTable.takeError());
  ASSERT_TRUE(bool(stateTable)) << llvm::toString(stateTable.takeError());
  const double baselineSimd =
      baselineTable->stages[0].implementations[0].totalCycles;
  const double stateSimd = stateTable->stages[0].implementations[0].totalCycles;
  EXPECT_DOUBLE_EQ(stateSimd - baselineSimd, 800.0 / 8.0);
}

TEST(StageCostModelTest,
     SuperBlockRecurrenceContentionStartsAbovePressureFreeFactor) {
  LogicalStage stage = logicalStage("stateful_recurrence",
                                    StageCostModelKind::LoopCarriedRecurrence,
                                    StageScheduleKind::LoopCarriedSerial, 16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.liveOutBytes = 4096;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.shuffleLaneSteps = 128.0;
  stage.workload.issueElements = 64.0;

  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_LE(costs[1].totalCycles, costs[0].totalCycles);
  EXPECT_LE(costs[2].totalCycles, costs[1].totalCycles);
  EXPECT_GE(costs[2].totalCycles,
            costs[2].resources.setup + 16.0 * costs[2].resources.issue);

  stage.workload.estimatedSpillTransactions = 32.0;
  auto spillingTable = evaluateOneStage(stage);
  ASSERT_TRUE(bool(spillingTable)) << llvm::toString(spillingTable.takeError());
  const auto &spillingCosts = spillingTable->stages.front().implementations;
  ASSERT_EQ(spillingCosts.size(), 3u);
  EXPECT_GT(spillingCosts[2].totalCycles, spillingCosts[1].totalCycles);
}

TEST(StageCostModelTest,
     MixedLocalStageSuperBlockHidesRecurrenceLatencyButKeepsIssueFloor) {
  LogicalStage stage = logicalStage(
      "mixed_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.localSimtMaterializable = true;
  stage.localSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.shuffleLaneSteps = 128.0;
  stage.workload.issueElements = 64.0;

  auto table = evaluateOneStage(stage, hardwareProfile());
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 4u);
  EXPECT_FALSE(costs[0].implementation.localScope);
  EXPECT_TRUE(costs[1].implementation.localScope);
  EXPECT_TRUE(costs[2].implementation.localScope);
  EXPECT_TRUE(costs[3].implementation.localScope);
  EXPECT_LE(costs[2].totalCycles, costs[1].totalCycles);
  EXPECT_LE(costs[3].totalCycles, costs[2].totalCycles);
  EXPECT_GE(costs[3].totalCycles,
            costs[3].resources.setup + 16.0 * costs[3].resources.issue);
}

TEST(StageCostModelTest,
     AtomicCostDoesNotRepriceOrdinaryLoadsAsIndirectMemory) {
  LogicalStage stage = logicalStage("atomic", StageCostModelKind::AtomicMemory,
                                    StageScheduleKind::PartiallyDependent);
  stage.features.hasAtomicMemory = true;
  stage.features.hasIndirectMemory = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  mlir::ascend::AtomicWorkload atomic;
  atomic.kind = "fadd";
  atomic.dataType = "f32";
  atomic.memorySemantic = "acq_rel";
  atomic.memoryScope = "gpu";
  atomic.logicalElements = 64.0;
  atomic.logicalOperationInstances = 1.0;
  atomic.contentionUnknown = true;
  stage.workload.atomicWorkloads.push_back(atomic);

  HardwareProfile profile = hardwareProfile();
  profile.simd.atomicRates["fadd.f32"] = {8.0, 2.0, 5.0, 3.0};
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  const auto &simd = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(simd.resources.load, 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 0.0);
  EXPECT_DOUBLE_EQ(simd.resources.atomic, 30.0);
}

TEST(StageCostModelTest, SuperBlockLatencyHidingStopsAtUsefulFactorLimit) {
  LogicalStage stage =
      logicalStage("simt_payload", StageCostModelKind::ScalarIssue);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.workload.loadWarpInstructions = 40.0;
  HardwareProfile cappedProfile = hardwareProfile();
  cappedProfile.superblockUsefulFactorLimit = 2;
  cappedProfile.superblockPersistentStatePressureFreeFactor = 2;
  auto table = evaluateOneStage(stage, cappedProfile);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  table->logicalProgramCountHint = 64;
  table->physicalCoreCountHint = 32;
  auto routes = solveStageRoutes(*table, cappedProfile.transition);
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 2);
  EXPECT_LT(routes->allSimt.totalCycles,
            2.0 * table->stages[0].implementations[0].totalCycles);
}

TEST(StageCostModelTest, PartialContinuousStageAlwaysUsesSerialCost) {
  LogicalStage stage =
      logicalStage("partial", StageCostModelKind::PartialContinuousTileMemory,
                   StageScheduleKind::IndependentPipelined);
  stage.features.hasPartialContinuousMemory = true;
  stage.features.hasIndirectMemory = true;
  stage.workload.operationElements.clear();
  stage.workload.loadBytes = 320.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadWarpInstructions = 10.0;
  stage.workload.storeWarpInstructions = 10.0;
  stage.workload.partialContinuousLoadRows = 10.0;
  stage.workload.partialContinuousStoreRows = 10.0;
  stage.workload.partialContinuousLoadBytes = 320.0;
  stage.workload.partialContinuousStoreBytes = 160.0;
  stage.workload.partialContinuousLoadWarpInstructions = 10.0;
  stage.workload.partialContinuousStoreWarpInstructions = 10.0;
  auto table = evaluateOneStage(stage);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  for (const StageImplementationCost &implementation :
       table->stages.front().implementations) {
    const auto &resource = implementation.resources;
    EXPECT_DOUBLE_EQ(implementation.totalCycles,
                     resource.setup + resource.load + resource.store);
  }
}
