// Tests for StageRouteCostModel responsibilities.
#include "CostModelTestUtils.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(StageRouteCostModelTest, KernelMixedRouteComesFromAdjacentStageModes) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto addStage = [&](llvm::StringRef id, double simd, double simt) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = id.str();
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    stage.implementations = {
        implementationCost({StageMode::SIMD, 1}, simd),
        implementationCost({StageMode::SIMT, 1}, simt),
        implementationCost({StageMode::SIMT, 1, true}, simt)};
    table.stages.push_back(stage);
  };
  addStage("head", 10.0, 20.0);
  addStage("payload", 100.0, 50.0);
  addStage("store", 30.0, 45.0);

  StageTransitionCost transition;
  transition.fixedPairCycles = 12.0;
  auto result = solveStageRoutes(table, transition);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_DOUBLE_EQ(result->allSimd.totalCycles, 140.0);
  EXPECT_DOUBLE_EQ(result->allSimt.totalCycles, 115.0);
  EXPECT_DOUBLE_EQ(result->mixed.totalCycles, 102.0);
  ASSERT_EQ(result->mixed.implementations.size(), 3u);
  EXPECT_EQ(result->mixed.implementations[0].mode, StageMode::SIMD);
  EXPECT_EQ(result->mixed.implementations[1].mode, StageMode::SIMT);
  EXPECT_EQ(result->mixed.implementations[2].mode, StageMode::SIMD);
  ASSERT_EQ(result->mixed.entryTransitionCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(result->mixed.entryTransitionCycles[1], 12.0);
}

TEST(StageRouteCostModelTest, MixedScopePaysExactBidirectionalUbHandoffCost) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.implementations = {implementationCost({StageMode::SIMD, 1}, 10.0),
                          implementationCost({StageMode::SIMT, 1}, 20.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "large_result_payload";
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1};
  payload.localSimtScopeCount = 2;
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 16384;
  payload.implementations = {
      implementationCost({StageMode::SIMD, 1}, 100.0),
      implementationCost({StageMode::SIMT, 1}, 10.0),
      implementationCost({StageMode::SIMT, 1, true}, 10.0)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  StageTransitionCost transition;
  transition.simdUbLoadBytesPerCycle = 512.0;
  transition.simdUbStoreBytesPerCycle = 256.0;
  transition.simtUbLoadBytesPerThreadPerCycle = 4.0;
  transition.simtUbStoreBytesPerThreadPerCycle = 4.0;
  transition.simtWarpSize = 32;
  auto routes = solveStageRoutes(table, transition);
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // Input: 4096/256 + 4096/(4*32) = 48 cycles.
  // Output: 16384/(4*32) + 16384/512 = 160 cycles.
  // Head/payload/tail: 10 + (10 + 208) + 10 = 238 cycles.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 238.0);
  ASSERT_EQ(routes->mixed.entryTransitionCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 208.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[2], 0.0);
  EXPECT_GT(routes->mixed.totalCycles, routes->allSimd.totalCycles);
}

TEST(StageRouteCostModelTest,
     MixedScopeSuperBlockAmortizesOnlyFixedTransitions) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";

  LogicalStageCost head;
  head.id = "head";
  head.implementations = {implementationCost({StageMode::SIMD, 1}, 10.0),
                          implementationCost({StageMode::SIMT, 1}, 100.0)};

  LogicalStageCost payload;
  payload.id = "payload";
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  payload.localSimtScopeCount = 1;
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 4096;
  payload.implementations = {
      implementationCost({StageMode::SIMD, 1}, 1000.0),
      implementationCost({StageMode::SIMT, 1}, 100.0),
      implementationCost({StageMode::SIMT, 1, true}, 100.0),
      implementationCost({StageMode::SIMT, 2, true}, 100.0),
      implementationCost({StageMode::SIMT, 4, true}, 100.0)};

  LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  StageTransitionCost transition;
  transition.fixedPairCycles = 80.0;
  transition.simdUbLoadBytesPerCycle = 512.0;
  transition.simdUbStoreBytesPerCycle = 256.0;
  transition.simtUbLoadBytesPerThreadPerCycle = 4.0;
  transition.simtUbStoreBytesPerThreadPerCycle = 4.0;
  transition.simtWarpSize = 32;
  auto routes = solveStageRoutes(table, transition);
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());

  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  // The 80-cycle fixed transition pair is amortized to 20 cycles.  The
  // 4096-byte input/output handoff remains 48 + 40 cycles per program.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles,
                   10.0 + 100.0 + 20.0 + 48.0 + 40.0 + 10.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 108.0);
}

TEST(StageRouteCostModelTest, MixedRouteRejectsUnmaterializableSimtStage) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.localSimtMaterializable = false;
  head.implementations = {implementationCost({StageMode::SIMD, 1}, 1.0),
                          implementationCost({StageMode::SIMT, 1}, 100.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "unmaterializable_payload";
  payload.localSimtMaterializable = false;
  payload.implementations = {implementationCost({StageMode::SIMD, 1}, 100.0),
                             implementationCost({StageMode::SIMT, 1}, 1.0)};
  table.stages = {head, payload};
  auto routes = solveStageRoutes(table, StageTransitionCost{});
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_FALSE(routes->mixed.legal);
}

TEST(StageRouteCostModelTest,
     MixedRouteReportsCheapestConstrainedRouteWhenLocalScopeLoses) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";

  mlir::ascend::LogicalStageCost gather;
  gather.id = "indirect_tile_gather";
  gather.localSimtMaterializable = true;
  gather.localSimtScopeCount = 1;
  gather.implementations = {
      implementationCost({StageMode::SIMD, 1}, 100.0),
      implementationCost({StageMode::SIMT, 1}, 130.0),
      implementationCost({StageMode::SIMT, 1, true}, 130.0)};
  mlir::ascend::LogicalStageCost dot;
  dot.id = "tiny_cube_dot";
  dot.implementations = {implementationCost({StageMode::SIMD, 1}, 40.0),
                         implementationCost({StageMode::SIMT, 1}, 90.0)};
  table.stages = {gather, dot};

  StageTransitionCost transition;
  transition.fixedPairCycles = 20.0;
  auto routes = solveStageRoutes(table, transition);
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  ASSERT_EQ(routes->mixed.implementations.size(), 2u);
  EXPECT_EQ(routes->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_TRUE(routes->mixed.implementations[0].localScope);
  EXPECT_EQ(routes->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 190.0);
}

TEST(StageRouteCostModelTest, AllSimdDoesNotPayRouteConditionalAutoBlockify) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";

  mlir::ascend::LogicalStageCost dispatch;
  dispatch.id = "physical_program_dispatch";
  dispatch.model = "auto_blockify_dispatch";
  dispatch.implementations = {implementationCost({StageMode::SIMD, 1}, 40.0),
                              implementationCost({StageMode::SIMT, 1}, 30.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "payload";
  payload.model = "scalar_issue";
  payload.implementations = {implementationCost({StageMode::SIMD, 1}, 100.0),
                             implementationCost({StageMode::SIMT, 1}, 80.0)};
  table.stages = {dispatch, payload};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  ASSERT_EQ(routes->allSimd.logicalStageCycles.size(), 2u);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 100.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 100.0);
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 110.0);
}

TEST(StageRouteCostModelTest, AllSimdPreservesSetupOnConditionalSchedule) {
  for (const auto *model : {"auto_blockify_dispatch", "auto_blockify_loop"}) {
    for (int64_t programs : {1, 5}) {
      StageCostTable table;
      table.profileVersion = "unit-test-profile-v1";
      table.logicalProgramCountHint = programs;
      table.physicalCoreCountHint = 2;
      mlir::ascend::LogicalStageCost dispatch;
      dispatch.id = "dispatch";
      dispatch.model = model;
      dispatch.implementations = {
          implementationCost({StageMode::SIMD, 1}, 40.0, 20.0),
          implementationCost({StageMode::SIMT, 1}, 30.0, 25.0)};
      mlir::ascend::LogicalStageCost payload;
      payload.id = "payload";
      payload.model = "scalar_issue";
      payload.implementations = {
          implementationCost({StageMode::SIMD, 1}, 100.0, 0.0),
          implementationCost({StageMode::SIMT, 1}, 80.0, 0.0)};
      table.stages = {dispatch, payload};
      auto routes = solveStageRoutes(table, StageTransitionCost{});
      ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
      const double waves = (programs + 1) / 2;
      ASSERT_TRUE(routes->allSimd.legal);
      EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[0], 20.0 * waves);
      EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 100.0 * waves);
      EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 120.0 * waves);
      ASSERT_TRUE(routes->allSimt.legal);
      EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 110.0 * waves);

      // Moving setup to the payload must not change the all-SIMD total.
      table.stages[0].implementations[0].resources.setup = 0.0;
      table.stages[0].implementations[0].totalCycles -= 20.0;
      table.stages[1].implementations[0].resources.setup = 20.0;
      table.stages[1].implementations[0].totalCycles += 20.0;
      auto moved = solveStageRoutes(table, StageTransitionCost{});
      ASSERT_TRUE(bool(moved)) << llvm::toString(moved.takeError());
      EXPECT_DOUBLE_EQ(moved->allSimd.totalCycles, routes->allSimd.totalCycles);
    }
  }
}

TEST(StageRouteCostModelTest, MixedRouteChargesEveryMaterializedScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.implementations = {implementationCost({StageMode::SIMD, 1}, 1.0),
                          implementationCost({StageMode::SIMT, 1}, 100.0)};
  mlir::ascend::LogicalStageCost gather;
  gather.id = "two_anchor_gather";
  gather.localSimtMaterializable = true;
  gather.localSimtFactors = {1};
  gather.localSimtScopeCount = 2;
  gather.implementations = {
      implementationCost({StageMode::SIMD, 1}, 100.0),
      implementationCost({StageMode::SIMT, 1}, 1.0),
      implementationCost({StageMode::SIMT, 1, true}, 1.0)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, gather, tail};

  StageTransitionCost transition;
  transition.fixedPairCycles = 20.0;
  auto routes = solveStageRoutes(table, transition);
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // 1 SIMD head + (10 enter + 1 payload + 20 extra scope pair) +
  // (10 leave + 1 SIMD tail).
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 43.0);
  ASSERT_EQ(routes->mixed.entryTransitionCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 40.0);
}

TEST(StageRouteCostModelTest, PureSimtRouteUsesOneUniformSuperBlockFactor) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  mlir::ascend::LogicalStageCost first;
  first.id = "first";
  first.implementations = {implementationCost({StageMode::SIMT, 1}, 5.0),
                           implementationCost({StageMode::SIMT, 2}, 1.0),
                           implementationCost({StageMode::SIMT, 4}, 3.0)};
  mlir::ascend::LogicalStageCost second;
  second.id = "second";
  second.implementations = {implementationCost({StageMode::SIMT, 1}, 5.0),
                            implementationCost({StageMode::SIMT, 2}, 4.0),
                            implementationCost({StageMode::SIMT, 4}, 1.0)};
  table.stages = {first, second};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  ASSERT_EQ(routes->allSimt.implementations.size(), 2u);
  EXPECT_EQ(routes->allSimt.implementations[0].superblockFactor, 4);
  EXPECT_EQ(routes->allSimt.implementations[1].superblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 4.0);
}

TEST(StageRouteCostModelTest, MixedScopeSuperBlockUsesSelectedFactorCost) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  mlir::ascend::LogicalStageCost prefix;
  prefix.id = "simd_prefix";
  prefix.implementations = {
      implementationCost({StageMode::SIMD, 1}, 5.0),
      implementationCost({StageMode::SIMT, 1}, 50.0),
      implementationCost({StageMode::SIMT, 2}, 25.0),
      implementationCost({StageMode::SIMT, 4}, 12.5),
      implementationCost({StageMode::SIMT, 1, true}, 50.0),
      implementationCost({StageMode::SIMT, 2, true}, 25.0),
      implementationCost({StageMode::SIMT, 4, true}, 12.5)};
  prefix.localSimtMaterializable = true;
  prefix.localSimtFactors = {1, 2, 4};

  mlir::ascend::LogicalStageCost payload;
  payload.id = "local_simt_payload";
  payload.implementations = {
      implementationCost({StageMode::SIMD, 1}, 100.0),
      implementationCost({StageMode::SIMT, 1}, 10.0),
      implementationCost({StageMode::SIMT, 2}, 1.0),
      implementationCost({StageMode::SIMT, 4}, 0.5),
      implementationCost({StageMode::SIMT, 1, true}, 10.0),
      implementationCost({StageMode::SIMT, 2, true}, 1.0),
      implementationCost({StageMode::SIMT, 4, true}, 0.5)};
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  table.stages = {prefix, payload};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 5.5);
}

TEST(StageRouteCostModelTest,
     FactoredMixedRouteUsesOneBackendMaterializableLocalScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";

  LogicalStageCost first;
  first.id = "first_local_candidate";
  first.features.replicatedByLocalSuperBlock = true;
  first.localSimtMaterializable = true;
  first.localSimtScopeCount = 1;
  first.implementations = {implementationCost({StageMode::SIMD, 1}, 100.0),
                           implementationCost({StageMode::SIMT, 4, true}, 1.0)};
  LogicalStageCost second = first;
  second.id = "second_local_candidate";
  LogicalStageCost tail;
  tail.id = "simd_tail";
  tail.implementations = {implementationCost({StageMode::SIMD, 1}, 1.0)};
  table.stages = {first, second, tail};

  auto routes = solveStageRoutes(table, StageTransitionCost{});
  ASSERT_TRUE(bool(routes)) << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(llvm::count_if(
                routes->mixed.implementations,
                [](const mlir::ascend::StageImplementation &implementation) {
                  return implementation.localScope;
                }),
            1);
  // One scope is SIMT (1 cycle); the other Stage remains SIMD and is cloned
  // once per grouped logical program (100 * F4); the outside tail is not.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 402.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[1], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.entryTransitionCycles[2], 0.0);
}
