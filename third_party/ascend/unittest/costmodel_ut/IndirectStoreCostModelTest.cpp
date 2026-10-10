// Tests for IndirectStoreCostModel responsibilities.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(IndirectStoreCostModelTest, RandomIndirectStoreStateAndResourceBoundary) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @probe(%p: tensor<32x!tt.ptr<f32>>, %v: tensor<32xf32>) {
        tt.store %p, %v : tensor<32x!tt.ptr<f32>>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto stage = logicalStage("store", StageCostModelKind::IndirectGatherMemory);
  module->walk([&](mlir::triton::StoreOp op) {
    stage.operations.push_back(op.getOperation());
  });
  auto &work = stage.workload;
  work.storeBytes = work.indirectStoreBytes = 128;
  work.storeWarpInstructions = work.indirectStoreTransactions = 1;
  work.scalarOperations = 7;
  work.loadBytes = 16;
  work.addressPatterns.push_back(loadedAddressPattern(stage, "tt.store", {32}));
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  auto old = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(old));
  profile.simdIndirectStoreModel = "random_f32_store_no_fill_first_20261007";
  profile.simtIndirectStoreModel = "random_f32_store_fill_ab_20261007";
  auto fit = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(fit));
  const auto &a = old->stages.front().implementations;
  const auto &b = fit->stages.front().implementations;
  EXPECT_DOUBLE_EQ(b[0].resources.store, 150.13769870695648 * 32);
  EXPECT_NEAR(b[1].resources.store, 145.04451206999323, 1e-9);
  for (unsigned i = 0; i < 2; ++i) {
    EXPECT_DOUBLE_EQ(b[i].resources.load, a[i].resources.load);
    EXPECT_DOUBLE_EQ(b[i].resources.scalar, a[i].resources.scalar);
    EXPECT_DOUBLE_EQ(b[i].resources.compute, a[i].resources.compute);
    EXPECT_EQ(b[i].indirectLoadPricing, "legacy_transactions");
    EXPECT_NE(b[i].indirectStorePricing, "legacy_transactions");
  }
  profile.simd.indirectDependencyLatencyCycles = 10000;
  auto changed = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(changed));
  EXPECT_DOUBLE_EQ(changed->stages.front().implementations[0].totalCycles,
                   b[0].totalCycles);
  profile.simdIndirectStoreModel = "random_f32_store_no_fill_reuse_20261007";
  changed = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(changed));
  EXPECT_EQ(changed->stages.front().implementations[0].indirectStorePricing,
            "legacy_transactions");
  work.hasProvenIndirectStoreReuse = true;
  changed = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(changed));
  EXPECT_DOUBLE_EQ(changed->stages.front().implementations[0].resources.store,
                   65.26772542192847 * 32);
  work.predicateElements = 1;
  changed = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(changed));
  for (const auto &cost : changed->stages.front().implementations)
    EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
  profile.simdIndirectStoreModel = "unknown";
  EXPECT_FALSE(profile.isValid());
  profile.simdIndirectStoreModel.clear();
  profile.target = "other-target";
  EXPECT_FALSE(profile.isValid());
}

TEST(IndirectStoreCostModelTest, RandomStoreFitAcceptsPartitionedTritonStore) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      tt.func public @scatter(%x: !tt.ptr<f32>, %idx: !tt.ptr<i32>,
                              %values: !tt.ptr<f32>) {
        %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
        %ip = tt.splat %idx : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
        %ip2 = tt.addptr %ip, %r : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
        %index = tt.load %ip2 : tensor<128x!tt.ptr<i32>>
        %vp = tt.splat %values : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %vp2 = tt.addptr %vp, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %value = tt.load %vp2 : tensor<128x!tt.ptr<f32>>
        %xp = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>> loc("scatter.py":15:12)
        %xp2 = tt.addptr %xp, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc("scatter.py":15:12)
        tt.store %xp2, %value : tensor<128x!tt.ptr<f32>> loc("scatter.py":15:15)
        tt.return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(
      *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition));
  const LogicalStage *payload = nullptr;
  for (const auto &stage : partition->stages)
    if (stage.workload.indirectStoreBytes > 0)
      payload = &stage;
  ASSERT_NE(payload, nullptr);
  ASSERT_EQ(payload->operations.size(), 3u);
  auto profile = hardwareProfile();
  profile.target = "Ascend950PR/dav-c310";
  profile.logicalWarpGroupCount = 1;
  auto old = evaluateOneStage(*payload, profile);
  ASSERT_TRUE(bool(old));
  profile.simdIndirectStoreModel = profile.simtIndirectStoreModel =
      "random_store_no_fill_first_20261008";
  auto fit = evaluateOneStage(*payload, profile);
  ASSERT_TRUE(bool(fit));
  const double expected[] = {125.41805018354371 + 162.87111975882544 * 128,
                             382.1012717872757 + 1.3935292896269613 * 128 -
                                 19.191779247532597 * 4};
  for (unsigned i = 0; i < 2; ++i) {
    const auto &cost = fit->stages.front().implementations[i];
    const auto &before = old->stages.front().implementations[i];
    EXPECT_EQ(cost.indirectStorePricing, profile.simdIndirectStoreModel);
    EXPECT_NEAR(cost.resources.store, expected[i], 1e-8);
    EXPECT_DOUBLE_EQ(cost.resources.load, before.resources.load);
    EXPECT_DOUBLE_EQ(cost.resources.compute, before.resources.compute);
    EXPECT_DOUBLE_EQ(cost.resources.scalar, before.resources.scalar);
    EXPECT_DOUBLE_EQ(cost.resources.issue, before.resources.issue);
    EXPECT_DOUBLE_EQ(cost.resources.synchronization,
                     before.resources.synchronization);
    EXPECT_NEAR(cost.totalCycles - before.totalCycles,
                expected[i] - before.resources.store, 1e-8);
  }
  auto multiple = *payload;
  multiple.operations.push_back(payload->operations.back());
  auto rejected = evaluateOneStage(multiple, profile);
  ASSERT_TRUE(bool(rejected));
  for (const auto &cost : rejected->stages.front().implementations)
    EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
}

TEST(IndirectStoreCostModelTest, RandomStoreDtypeRankStateAndDomain) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto p = hardwareProfile();
  p.target = "Ascend950PR/dav-c310";
  for (int bits : {1, 8, 16, 32, 64}) {
    for (int e : {8, 128, 256, 512, 2048}) {
      for (int w : {1, 2, 4, 8, 16, 32, 64}) {
        for (bool reuse : {false, true}) {
          // Rank 1, 3 and 8 exercise shape-independent pricing. High ranks
          // rely on per-axis loaded evidence, not the legacy rank-5 predicate.
          for (bool rank3 : {false, true}) {
            llvm::SmallVector<int64_t> shape =
                rank3 ? llvm::SmallVector<int64_t>{2, 2, e / 4}
                      : llvm::SmallVector<int64_t>{e};
            if (rank3 && e >= 256)
              shape = {2, 2, 2, 2, 2, 2, 2, e / 128};
            auto type = mlir::RankedTensorType::get(
                shape, mlir::IntegerType::get(&context, bits));
            mlir::Block block;
            auto loc = mlir::UnknownLoc::get(&context);
            auto ptr = block.addArgument(mlir::IndexType::get(&context), loc);
            auto value = block.addArgument(type, loc);
            mlir::OperationState state(loc, "tt.store");
            state.addOperands({ptr, value});
            auto *op = mlir::Operation::create(state);
            LogicalStage s;
            s.id = "random-store";
            s.costModelKind = StageCostModelKind::IndirectGatherMemory;
            s.simdLegal = s.simtLegal = true;
            s.legalSimtFactors = {1};
            s.operations = {op};
            auto &a = s.workload;
            a.storeBytes = a.indirectStoreBytes = std::max(1, bits / 8) * e;
            a.storeWarpInstructions = a.indirectStoreTransactions = 1;
            a.hasProvenIndirectStoreReuse = reuse;
            a.addressPatterns = {loadedAddressPattern(
                s, "tt.store", shape, "opaque_loaded", type.getRank() <= 5)};
            p.logicalWarpGroupCount = w;
            p.simdIndirectStoreModel = p.simtIndirectStoreModel =
                reuse ? "random_store_no_fill_reuse_20261008"
                      : "random_store_no_fill_first_20261008";
            auto fit = evaluateImplementations(s, p);
            int g = bits <= 16 ? 0 : (bits == 32 ? 1 : 2);
            const double beta[2][3] = {
                {152.8758408085307, 162.87111975882544, 161.07674811788297},
                {68.58965840703893, 69.21236829277788, 64.45492494749357}};
            const double b[2][3] = {
                {4.664669494263474, 1.3935292896269613, 2.0019961511928295},
                {4.4638287665056, 1.1754145133075695, 2.076648173015399}};
            const double d[2][3] = {
                {-25.526112727120946, -19.191779247532597, -35.604138981973364},
                {-9.072748388478542, -20.88472709201174, -38.4225252759396}};
            bool supported[] = {w == 1,
                                e <= 512 * w && (w == 1 || e >= 32 * w)};
            double expected[] = {
                (reuse ? 0 : 125.41805018354371) + beta[reuse][g] * e,
                (reuse ? 70.0956884939018 : 382.1012717872757) +
                    b[reuse][g] * e +
                    (reuse ? .8774560851959086 : .6797627278581438) *
                        std::max(e - (reuse ? 128 : 256), 0) +
                    d[reuse][g] * e / (32.0 * w)};
            for (unsigned i = 0; i < 2; ++i) {
              if (supported[i]) {
                EXPECT_EQ(fit[i].indirectStorePricing,
                          p.simdIndirectStoreModel);
                EXPECT_NEAR(fit[i].resources.store, expected[i], 1e-7);
              } else
                EXPECT_EQ(fit[i].indirectStorePricing, "legacy_transactions");
            }
            if (reuse) {
              a.hasProvenIndirectStoreReuse = false;
              for (const auto &cost : evaluateImplementations(s, p))
                EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
              a.hasProvenIndirectStoreReuse = true;
            }
            a.addressPatterns[0].axes[0].regularity = "fixed_stride";
            for (const auto &cost : evaluateImplementations(s, p))
              EXPECT_EQ(cost.indirectStorePricing, "legacy_transactions");
            op->destroy();
          }
        }
      }
    }
  }
}
