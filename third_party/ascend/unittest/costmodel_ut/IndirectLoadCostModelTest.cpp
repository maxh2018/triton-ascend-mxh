// Tests for IndirectLoadCostModel responsibilities.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"
#include "StageIRTestUtils.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(IndirectLoadCostModelTest,
     SimdMatchedIndirectFitReplacesOnlyLoadResource) {
  IRTestContext context(true, false);
  auto module = context.parse(R"mlir(
    module {
      func.func @probe(%p: tensor<32x!tt.ptr<f32>>) {
        %x = tt.load %p : tensor<32x!tt.ptr<f32>>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory);
  module->walk([&](mlir::triton::LoadOp load) {
    stage.operations.push_back(load.getOperation());
  });
  auto &work = stage.workload;
  work.loadBytes = work.indirectLoadBytes = 128;
  work.loadWarpInstructions = work.indirectLoadTransactions = 32;
  work.scalarOperations = 7;
  work.storeBytes = 16;
  work.addressPatterns.push_back(loadedAddressPattern(stage, "tt.load", {32}));
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  auto old = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(old));
  profile.simdIndirectLoadModel = "random_f32_matched_ab_20261007";
  auto fit = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(fit));
  const auto &a = old->stages.front().implementations;
  const auto &b = fit->stages.front().implementations;
  EXPECT_DOUBLE_EQ(b[0].resources.load, 83.56748010753823 * 32);
  EXPECT_DOUBLE_EQ(b[0].resources.scalar, a[0].resources.scalar);
  EXPECT_DOUBLE_EQ(b[0].resources.store, a[0].resources.store);
  EXPECT_DOUBLE_EQ(b[0].resources.compute, a[0].resources.compute);
  EXPECT_DOUBLE_EQ(b[1].totalCycles, a[1].totalCycles);
  profile.simd.indirectDependencyLatencyCycles = 10000;
  auto changed = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(changed));
  EXPECT_DOUBLE_EQ(changed->stages.front().implementations[0].totalCycles,
                   b[0].totalCycles);
}

TEST(IndirectLoadCostModelTest, SimdMatchedIndirectNarrowRankTwoDomain) {
  IRTestContext context(true, false);
  auto module = context.parse(R"mlir(
    module {
      func.func @probe(%p: tensor<4x8x!tt.ptr<f32>>) {
        %x = tt.load %p : tensor<4x8x!tt.ptr<f32>>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto stage = logicalStage("rank2", StageCostModelKind::IndirectGatherMemory);
  module->walk([&](mlir::triton::LoadOp load) {
    stage.operations.push_back(load.getOperation());
  });
  auto &work = stage.workload;
  work.loadBytes = work.indirectLoadBytes = 128;
  work.loadWarpInstructions = work.indirectLoadTransactions = 32;
  mlir::ascend::AddressPatternSummary pattern;
  pattern.memoryOp = "tt.load";
  pattern.stageId = stage.id;
  pattern.dependsOnLoadedValue = true;
  mlir::ascend::AddressAxisSummary outer, inner;
  outer.extent = 4;
  outer.regularity = "fixed_stride";
  outer.knownStride = 4096;
  inner.extent = 8;
  inner.regularity = "opaque_loaded";
  pattern.axes = {outer, inner};
  work.addressPatterns = {pattern};
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  profile.simdIndirectLoadModel = "random_f32_matched_ab_20261007";
  for (const char *category : {"fixed_stride", "opaque_loaded"}) {
    work.addressPatterns[0].axes[0].regularity = category;
    auto result = evaluateOneStage(stage, profile);
    ASSERT_TRUE(bool(result));
    EXPECT_DOUBLE_EQ(result->stages.front().implementations[0].resources.load,
                     83.56748010753823 * 32);
  }
  work.addressPatterns[0].axes[0].regularity = "computed_nonaffine";
  auto fallback = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(fallback));
  EXPECT_EQ(fallback->stages.front().implementations[0].indirectLoadPricing,
            "legacy_transactions");
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankFitPreservesIndependentResources) {
  IRTestContext context(false, true);
  const llvm::SmallVector<mlir::Type> types = {
      mlir::IntegerType::get(&context, 8),
      mlir::IntegerType::get(&context, 16),
      mlir::IntegerType::get(&context, 32),
      mlir::IntegerType::get(&context, 64),
      mlir::Float16Type::get(&context),
      mlir::BFloat16Type::get(&context),
      mlir::Float32Type::get(&context),
      mlir::Float8E4M3FNType::get(&context),
      mlir::Float8E5M2Type::get(&context)};
  const llvm::SmallVector<llvm::SmallVector<int64_t>> shapes = {
      {4},     {8},     {16},       {32},         {2048},
      {8, 16}, {8, 64}, {2, 4, 16}, {2, 2, 4, 8}, {2, 2, 2, 4, 8}};
  for (mlir::Type element : types) {
    for (const auto &shape : shapes) {
      auto type = mlir::RankedTensorType::get(shape, element);
      const int64_t elements = type.getNumElements();
      mlir::Block input;
      auto ptr = input.addArgument(mlir::IndexType::get(&context),
                                   mlir::UnknownLoc::get(&context));
      mlir::OperationState state(mlir::UnknownLoc::get(&context), "tt.load");
      state.addOperands(ptr);
      state.addTypes(type);
      auto *op = mlir::Operation::create(state);
      auto stage =
          logicalStage("dtype-rank", StageCostModelKind::IndirectGatherMemory,
                       StageScheduleKind::StraightLine, 3);
      stage.operations = {op};
      stage.features.loopBackedgeCount = 1;
      auto &work = stage.workload;
      work.loadBytes = work.indirectLoadBytes =
          elements * (element.getIntOrFloatBitWidth() / 8);
      work.loadWarpInstructions = work.indirectLoadTransactions = elements;
      work.scalarOperations = 7;
      work.storeBytes = 16;
      work.storeWarpInstructions = 2;
      work.addressPatterns = {
          loadedAddressPattern(stage, "tt.load", shape, "opaque")};
      auto profile = hardwareProfile();
      profile.target = "Ascend950PR/dav-c310";
      auto legacy = evaluateOneStage(stage, profile);
      ASSERT_TRUE(bool(legacy));
      profile.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
      auto calibrated = evaluateOneStage(stage, profile);
      ASSERT_TRUE(bool(calibrated));
      const auto &old = legacy->stages.front().implementations;
      const auto &fit = calibrated->stages.front().implementations;
      const bool small32 = element.getIntOrFloatBitWidth() == 32 &&
                           shape.size() == 1 && elements <= 16;
      EXPECT_EQ(fit[0].indirectLoadPricing, profile.simdIndirectLoadModel);
      EXPECT_DOUBLE_EQ(fit[0].resources.load,
                       elements *
                           (small32 ? 49.57621548794853 : 81.94869549595556));
      EXPECT_DOUBLE_EQ(fit[0].resources.scalar, old[0].resources.scalar);
      EXPECT_DOUBLE_EQ(fit[0].resources.compute, old[0].resources.compute);
      EXPECT_DOUBLE_EQ(fit[0].resources.store, old[0].resources.store);
      EXPECT_DOUBLE_EQ(fit[0].resources.loopControl,
                       old[0].resources.loopControl);
      EXPECT_DOUBLE_EQ(fit[0].resources.setup, old[0].resources.setup);
      EXPECT_NEAR(fit[0].totalCycles - old[0].totalCycles,
                  3 * (fit[0].resources.load - old[0].resources.load), 1e-8);
      EXPECT_DOUBLE_EQ(fit[1].totalCycles, old[1].totalCycles);
      profile.simd.indirectDependencyLatencyCycles = 10000;
      for (int64_t warps : {1, 2, 4, 8, 16, 32, 64}) {
        profile.logicalWarpGroupCount = warps;
        auto result = evaluateOneStage(stage, profile);
        ASSERT_TRUE(bool(result));
        EXPECT_DOUBLE_EQ(result->stages.front().implementations[0].totalCycles,
                         fit[0].totalCycles);
      }
      work.predicateElements = 1;
      auto masked = evaluateOneStage(stage, profile);
      ASSERT_TRUE(bool(masked));
      EXPECT_EQ(masked->stages.front().implementations[0].indirectLoadPricing,
                "legacy_transactions");
      work.predicateElements = 0;
      work.estimatedSpillTransactions = 1;
      auto spilled = evaluateOneStage(stage, profile);
      ASSERT_TRUE(bool(spilled));
      EXPECT_EQ(spilled->stages.front().implementations[0].indirectLoadPricing,
                "legacy_transactions");
      op->destroy();
    }
  }
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankFitRejectsUnvalidatedDomains) {
  IRTestContext context(false, true);
  const llvm::SmallVector<std::pair<mlir::Type, llvm::SmallVector<int64_t>>>
      cases = {{mlir::Float64Type::get(&context), {32}},
               {mlir::IntegerType::get(&context, 1), {32}},
               {mlir::Float8E4M3FNUZType::get(&context), {32}},
               {mlir::IntegerType::get(&context, 32), {2}},
               {mlir::IntegerType::get(&context, 32), {4096}},
               {mlir::IntegerType::get(&context, 32), {1, 32}},
               {mlir::IntegerType::get(&context, 32), {4, 2}},
               {mlir::IntegerType::get(&context, 32), {2, 2, 2, 2, 2, 2}}};
  for (const auto &test : cases) {
    auto type = mlir::RankedTensorType::get(test.second, test.first);
    mlir::Block input;
    auto ptr = input.addArgument(mlir::IndexType::get(&context),
                                 mlir::UnknownLoc::get(&context));
    mlir::OperationState state(mlir::UnknownLoc::get(&context), "tt.load");
    state.addOperands(ptr);
    state.addTypes(type);
    auto *op = mlir::Operation::create(state);
    auto stage =
        logicalStage("fallback", StageCostModelKind::IndirectGatherMemory);
    stage.operations = {op};
    auto &work = stage.workload;
    work.loadBytes = work.indirectLoadBytes =
        (type.getNumElements() * test.first.getIntOrFloatBitWidth() + 7) / 8;
    work.loadWarpInstructions = work.indirectLoadTransactions =
        type.getNumElements();
    work.addressPatterns = {
        loadedAddressPattern(stage, "tt.load", test.second)};
    auto profile = hardwareProfile();
    profile.target = "Ascend950PR/dav-c310";
    profile.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
    auto result = evaluateOneStage(stage, profile);
    ASSERT_TRUE(bool(result));
    EXPECT_EQ(result->stages.front().implementations[0].indirectLoadPricing,
              "legacy_transactions");
    op->destroy();
  }
}

TEST(IndirectLoadCostModelTest, SimdDtypeRankFitAcceptsPartitionedTritonLoad) {
  IRTestContext context(true, false);
  auto module = context.parse(R"mlir(
    module {
      tt.func public @probe(%x: !tt.ptr<i16>, %idx: !tt.ptr<i32>, %out: !tt.ptr<i16>) {
        %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
        %ip = tt.splat %idx : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
        %ip2 = tt.addptr %ip, %r : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
        %index = tt.load %ip2 : tensor<32x!tt.ptr<i32>>
        %xp = tt.splat %x : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>>
        %xp2 = tt.addptr %xp, %index : tensor<32x!tt.ptr<i16>>, tensor<32xi32>
        %payload = tt.load %xp2 : tensor<32x!tt.ptr<i16>>
        %yp = tt.splat %out : !tt.ptr<i16> -> tensor<32x!tt.ptr<i16>>
        %yp2 = tt.addptr %yp, %r : tensor<32x!tt.ptr<i16>>, tensor<32xi32>
        tt.store %yp2, %payload : tensor<32x!tt.ptr<i16>>
        tt.return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(
      *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition));
  const LogicalStage *payload = nullptr;
  for (const auto &stage : partition->stages)
    if (stage.costModelKind == StageCostModelKind::IndirectGatherMemory)
      payload = &stage;
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(llvm::count_if(payload->operations,
                           [](mlir::Operation *op) {
                             return op->getName().getStringRef() == "tt.load";
                           }),
            1);
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  profile.simdIndirectLoadModel = "random_dtype_matched_ab_20261008";
  auto result = evaluateOneStage(*payload, profile);
  ASSERT_TRUE(bool(result));
  const auto &fit = result->stages.front().implementations[0];
  EXPECT_EQ(fit.indirectLoadPricing, profile.simdIndirectLoadModel);
  EXPECT_DOUBLE_EQ(fit.resources.load, 81.94869549595556 * 32);
  // The same partitioned load must also admit SIMT dtype pricing.
  profile.simtIndirectLoadModel = "random_dtype_six_term_20261008";
  auto simtResult = evaluateOneStage(*payload, profile);
  ASSERT_TRUE(bool(simtResult));
  const auto &simt = simtResult->stages.front().implementations[1];
  EXPECT_EQ(simt.indirectLoadPricing, profile.simtIndirectLoadModel);
  EXPECT_NEAR(simt.resources.load,
              12.456944150614481 + 1.6732132663368477 * 32 + 95.13859585355112 +
                  0.11214618741568584 * 32,
              1e-10);
}

TEST(IndirectLoadCostModelTest,
     SimtDtypeRankFitUsesStorageWidthAndPreservesResources) {
  IRTestContext context(false, true);
  const llvm::SmallVector<mlir::Type> types = {
      mlir::IntegerType::get(&context, 8),
      mlir::IntegerType::get(&context, 16),
      mlir::IntegerType::get(&context, 32),
      mlir::IntegerType::get(&context, 64),
      mlir::IntegerType::get(&context, 32, mlir::IntegerType::Unsigned),
      mlir::Float16Type::get(&context),
      mlir::BFloat16Type::get(&context),
      mlir::Float32Type::get(&context),
      mlir::Float64Type::get(&context),
      mlir::Float8E4M3FNType::get(&context),
      mlir::Float8E5M2Type::get(&context)};
  // Ordering: alpha, beta, sigma, phi, gamma, nu (same as the frozen model).
  const double coefficients[][6] = {
      {7.617911783542814, 1.5780971099377332, 92.18952965944916,
       0.19387737354356369, 0, 0.049471659398567715},
      {12.456944150614481, 1.6732132663368477, 95.13859585355112,
       0.11214618741568584, 5.634318857353958, 0.026588037490546692},
      {50.71936539506111, 1.693396365504775, 60.38681262899275,
       0.09265123974451413, 82.91193957489877, 0.026814982781878355},
      {96.09572807125711, 1.937402636239715, 24.962585867214496,
       0.20099293498091206, 41.8669299352394, 0.03855485734655757}};
  for (mlir::Type element : types) {
    const int64_t bytes = element.getIntOrFloatBitWidth() / 8;
    const auto &beta = coefficients[bytes == 1   ? 0
                                    : bytes == 2 ? 1
                                    : bytes == 4 ? 2
                                                 : 3];
    for (int64_t warps : {1, 2, 4, 8, 16, 32, 64}) {
      for (int64_t q : {1, 2, 4}) {
        const int64_t elements = 32 * warps * q;
        for (int64_t rank = 1; rank <= 5; ++rank) {
          llvm::SmallVector<int64_t> shape(rank, 2);
          if (rank == 1) {
            shape[0] = elements;
          } else {
            shape.back() =
                std::min<int64_t>(32, elements / (1LL << (rank - 1)));
            shape[rank - 2] = elements / (shape.back() * (1LL << (rank - 2)));
          }
          SCOPED_TRACE(element.getIntOrFloatBitWidth());
          SCOPED_TRACE(warps);
          SCOPED_TRACE(q);
          SCOPED_TRACE(rank);
          mlir::Block input;
          auto ptr = input.addArgument(mlir::IndexType::get(&context),
                                       mlir::UnknownLoc::get(&context));
          mlir::OperationState state(mlir::UnknownLoc::get(&context),
                                     "tt.load");
          state.addOperands(ptr);
          state.addTypes(mlir::RankedTensorType::get(shape, element));
          auto *load = mlir::Operation::create(state);
          mlir::OperationState helperState(mlir::UnknownLoc::get(&context),
                                           "test.address");
          auto *helper = mlir::Operation::create(helperState);
          auto stage = logicalStage("simt-width",
                                    StageCostModelKind::IndirectGatherMemory,
                                    StageScheduleKind::StraightLine, 3);
          stage.operations = {helper, load};
          mlir::Operation *regionOwner = nullptr;
          if (rank >= 3) {
            mlir::OperationState ownerState(mlir::UnknownLoc::get(&context),
                                            "test.region");
            ownerState.addRegion();
            regionOwner = mlir::Operation::create(ownerState);
            auto &block = regionOwner->getRegion(0).emplaceBlock();
            block.push_back(helper);
            block.push_back(load);
            stage.operations = {regionOwner};
          }
          stage.features.loopBackedgeCount = 1;
          auto &work = stage.workload;
          work.loadBytes = work.indirectLoadBytes = elements * bytes;
          work.loadWarpInstructions = work.indirectLoadTransactions =
              elements / 32;
          work.scalarOperations = 7;
          work.storeBytes = 16;
          work.storeWarpInstructions = 2;
          work.addressPatterns = {
              loadedAddressPattern(stage, "tt.load", shape)};
          auto profile = hardwareProfile();
          profile.target = "Ascend950PR/dav-c310";
          profile.logicalWarpGroupCount = warps;
          auto old = evaluateOneStage(stage, profile);
          ASSERT_TRUE(bool(old));
          profile.simtIndirectLoadModel = "random_dtype_six_term_20261008";
          auto result = evaluateOneStage(stage, profile);
          ASSERT_TRUE(bool(result));
          const auto &fit = result->stages.front().implementations[1];
          const auto &legacy = old->stages.front().implementations[1];
          const double columns = rank == 1 ? 32 : shape.back();
          const double expected =
              beta[0] + beta[1] * elements + beta[2] * (elements <= 128) +
              beta[3] * elements * (rank == 1) +
              beta[4] * std::max(4.0 * q / columns - 2, 0.0) +
              beta[5] * elements * std::max(std::log2(32.0 / columns), 0.0);
          EXPECT_EQ(fit.indirectLoadPricing, profile.simtIndirectLoadModel);
          EXPECT_NEAR(fit.resources.load, expected, 1e-9);
          EXPECT_DOUBLE_EQ(fit.resources.scalar, legacy.resources.scalar);
          EXPECT_DOUBLE_EQ(fit.resources.compute, legacy.resources.compute);
          EXPECT_DOUBLE_EQ(fit.resources.store, legacy.resources.store);
          EXPECT_DOUBLE_EQ(fit.resources.setup, legacy.resources.setup);
          EXPECT_DOUBLE_EQ(fit.resources.loopControl,
                           legacy.resources.loopControl);
          EXPECT_DOUBLE_EQ(fit.resources.issue, legacy.resources.issue);
          EXPECT_NEAR(fit.totalCycles - legacy.totalCycles,
                      3 * (expected - legacy.resources.load), 1e-8);
          EXPECT_DOUBLE_EQ(
              result->stages.front().implementations[0].totalCycles,
              old->stages.front().implementations[0].totalCycles);
          profile.simt.indirectDependencyLatencyCycles = 10000;
          auto noDoubleCharge = evaluateOneStage(stage, profile);
          ASSERT_TRUE(bool(noDoubleCharge));
          EXPECT_DOUBLE_EQ(
              noDoubleCharge->stages.front().implementations[1].totalCycles,
              fit.totalCycles);
          if (regionOwner) {
            regionOwner->destroy();
          } else {
            load->destroy();
            helper->destroy();
          }
        }
      }
    }
  }
}

TEST(IndirectLoadCostModelTest, SimtDtypeRankFitRejectsUnvalidatedDomains) {
  IRTestContext context(false, true);
  const llvm::SmallVector<llvm::SmallVector<int64_t>> shapes = {
      {32}, {16}, {2, 128, 2}, {2, 2, 2, 2, 2, 2}, {1, 32}, {2, 256}};
  for (size_t index = 0; index < shapes.size(); ++index) {
    auto type = mlir::RankedTensorType::get(
        shapes[index], mlir::IntegerType::get(&context, 32));
    mlir::Block input;
    auto ptr = input.addArgument(mlir::IndexType::get(&context),
                                 mlir::UnknownLoc::get(&context));
    mlir::OperationState state(mlir::UnknownLoc::get(&context), "tt.load");
    state.addOperands(ptr);
    state.addTypes(type);
    auto *load = mlir::Operation::create(state);
    auto stage =
        logicalStage("guard", StageCostModelKind::IndirectGatherMemory);
    stage.operations = {load};
    stage.workload.loadBytes = stage.workload.indirectLoadBytes =
        type.getNumElements() * 4;
    stage.workload.loadWarpInstructions =
        stage.workload.indirectLoadTransactions = 1;
    stage.workload.addressPatterns = {
        loadedAddressPattern(stage, "tt.load", shapes[index])};
    auto profile = hardwareProfile();
    profile.target = "Ascend950PR/dav-c310";
    profile.logicalWarpGroupCount = (index == 2 || index == 5) ? 4 : 1;
    profile.simtIndirectLoadModel = "random_dtype_six_term_20261008";
    auto result = evaluateOneStage(stage, profile);
    ASSERT_TRUE(bool(result));
    EXPECT_EQ(result->stages.front().implementations[1].indirectLoadPricing,
              index == 0 ? profile.simtIndirectLoadModel
                         : "legacy_transactions");
    if (index == 0) {
      for (unsigned guard = 0; guard < 8; ++guard) {
        auto invalid = stage;
        mlir::Operation *secondLoad = nullptr;
        if (guard == 0)
          invalid.workload.predicateElements = 1;
        if (guard == 1)
          invalid.workload.estimatedSpillTransactions = 1;
        if (guard == 2)
          invalid.features.activeLaneRatio = 0.5;
        if (guard == 3) {
          invalid.workload.storeBytes = invalid.workload.indirectStoreBytes = 4;
        }
        if (guard == 4)
          invalid.workload.addressPatterns[0].axes[0].regularity =
              "fixed_stride";
        if (guard == 5)
          invalid.workload.addressPatterns[0].dependsOnLoadedValue = false;
        if (guard == 6) {
          secondLoad = load->clone();
          invalid.operations.push_back(secondLoad);
        }
        if (guard == 7)
          invalid.features.hasAtomicMemory = true;
        auto rejected = evaluateOneStage(invalid, profile);
        ASSERT_TRUE(bool(rejected));
        EXPECT_EQ(
            rejected->stages.front().implementations[1].indirectLoadPricing,
            "legacy_transactions");
        if (secondLoad)
          secondLoad->destroy();
      }
      profile.simtIndirectLoadModel = "unknown_load_fit";
      EXPECT_FALSE(profile.isValid());
    }
    load->destroy();
  }
}

TEST(IndirectLoadCostModelTest, RandomIndirectFitReplacesOnlyLoadResource) {
  IRTestContext context(true, false);
  auto module = context.parse(R"mlir(
    module {
      func.func @probe(%p: tensor<256x!tt.ptr<i32>>) {
        %x = tt.load %p : tensor<256x!tt.ptr<i32>>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::StraightLine, 3);
  module->walk([&](mlir::triton::LoadOp load) {
    stage.operations.push_back(load.getOperation());
  });
  auto &work = stage.workload;
  work.loadBytes = work.indirectLoadBytes = 1024;
  work.loadWarpInstructions = work.indirectLoadTransactions = 8;
  work.scalarOperations = 7;
  work.storeWarpInstructions = 2;
  work.storeBytes = 16;
  work.addressPatterns.push_back(loadedAddressPattern(stage, "tt.load", {256}));
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  profile.logicalWarpGroupCount = 4;
  auto legacy = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(legacy));
  profile.simtIndirectLoadModel = "random_i32_six_term_20261007";
  auto calibrated = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(calibrated));
  const auto &oldCosts = legacy->stages.front().implementations;
  const auto &newCosts = calibrated->stages.front().implementations;
  EXPECT_DOUBLE_EQ(newCosts[0].totalCycles, oldCosts[0].totalCycles);
  const double fitted =
      62.930686069640686 + 1.5502588407166296 * 256 + 0.19338028618547126 * 256;
  EXPECT_DOUBLE_EQ(newCosts[1].resources.load, fitted);
  EXPECT_DOUBLE_EQ(newCosts[1].resources.scalar, oldCosts[1].resources.scalar);
  EXPECT_DOUBLE_EQ(newCosts[1].resources.store, oldCosts[1].resources.store);
  EXPECT_DOUBLE_EQ(newCosts[1].resources.compute,
                   oldCosts[1].resources.compute);
  EXPECT_NEAR(newCosts[1].totalCycles - oldCosts[1].totalCycles,
              3 * (fitted - oldCosts[1].resources.load), 1e-9);
  // Changing the old dependency latency cannot affect a fitted load.
  profile.simt.indirectDependencyLatencyCycles = 10000;
  auto changed = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(changed));
  EXPECT_DOUBLE_EQ(changed->stages.front().implementations[1].totalCycles,
                   newCosts[1].totalCycles);
  // Unknown/non-affine axes and spill remain outside the calibrated domain.
  profile.simt.indirectDependencyLatencyCycles = 20;
  stage.workload.addressPatterns.front().axes.front().regularity =
      "computed_nonaffine";
  auto fallback = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(fallback));
  EXPECT_DOUBLE_EQ(fallback->stages.front().implementations[1].totalCycles,
                   oldCosts[1].totalCycles);
}

TEST(IndirectLoadCostModelTest, IndirectMemoryUsesDependencyProfile) {
  LogicalStage stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadBytes = 1024.0;
  stage.workload.indirectLoadTransactions = 8.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.indirectLoadTransactionsPerCycle = 0.25;
  profile.simd.indirectDependencyLatencyCycles = 80.0;
  profile.simt.indirectLoadTransactionsPerCycle = 1.0;
  profile.simt.indirectDependencyLatencyCycles = 20.0;
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 112.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 28.0);
  EXPECT_LT(costs[1].totalCycles, costs[0].totalCycles);
}
