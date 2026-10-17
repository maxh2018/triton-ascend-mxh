#include "AscendModel/Analysis/StagePartitioner.h"
#include "IndirectMemoryTestUtils.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::test;

namespace {
// Real Triton ops exercise reshape/prefix proofs as well as formula dispatch.
OwningOpRef<ModuleOp> maskModule(IRTestContext &context,
                                 ArrayRef<int64_t> shape, int64_t k, bool store,
                                 bool random = false, StringRef dtype = "f32") {
  int64_t e = 1;
  std::string dims;
  for (int64_t d : shape) {
    e *= d;
    dims += std::to_string(d) + "x";
  }
  std::string source = llvm::formatv(R"mlir(
    module {{ tt.func @masked(%base: !tt.ptr<f32>,
        %indices_base: !tt.ptr<i32>, %masks_base: !tt.ptr<i8>,
        %values: tensor<{1}f32>) {{
      %input_lane = tt.make_range {{start = 0 : i32, end = {0} : i32} : tensor<{0}xi32>
      %indices_splat = tt.splat %indices_base : !tt.ptr<i32> -> tensor<{0}x!tt.ptr<i32>>
      %indices = tt.addptr %indices_splat, %input_lane : tensor<{0}x!tt.ptr<i32>>, tensor<{0}xi32>
      %i = tt.load %indices : tensor<{0}x!tt.ptr<i32>>
      %offset = tt.reshape %i : tensor<{0}xi32> -> tensor<{1}i32>
      %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<{1}!tt.ptr<f32>>
      %ptr = tt.addptr %base_tensor, %offset : tensor<{1}!tt.ptr<f32>>, tensor<{1}i32>
  )mlir",
                                     e, dims)
                           .str();
  source += random ? llvm::formatv(R"mlir(
      %masks_splat = tt.splat %masks_base : !tt.ptr<i8> -> tensor<{0}x!tt.ptr<i8>>
      %masks = tt.addptr %masks_splat, %input_lane : tensor<{0}x!tt.ptr<i8>>, tensor<{0}xi32>
      %m = tt.load %masks : tensor<{0}x!tt.ptr<i8>>
      %zero_mask = arith.constant dense<0> : tensor<{0}xi8>
      %flat = arith.cmpi ne, %m, %zero_mask : tensor<{0}xi8>
  )mlir",
                                   e)
                         .str()
                   : llvm::formatv(R"mlir(
      %lane = tt.make_range {{start = 0 : i32, end = {0} : i32} : tensor<{0}xi32>
      %limit = arith.constant dense<{1}> : tensor<{0}xi32>
      %flat = arith.cmpi slt, %lane, %limit : tensor<{0}xi32>
  )mlir",
                                   e, k)
                         .str();
  source += llvm::formatv(
                "%mask = tt.reshape %flat : tensor<{0}xi1> -> tensor<{1}i1>\n",
                e, dims)
                .str();
  source +=
      store ? llvm::formatv(
                  "tt.store %ptr, %values, %mask : tensor<{0}!tt.ptr<f32>>\n",
                  dims)
                  .str()
            : llvm::formatv(
                  "%zero = arith.constant dense<0.0> : tensor<{0}f32>\n%out = "
                  "tt.load %ptr, %mask, %zero : tensor<{0}!tt.ptr<f32>>\n",
                  dims)
                  .str();
  source += "tt.return } }";
  if (dtype != "f32") {
    const size_t zero = source.find("dense<0.0>");
    if (zero != std::string::npos)
      source.replace(zero, 10, "dense<0>");
  }
  for (size_t at = 0; (at = source.find("f32", at)) != std::string::npos;
       at += dtype.size())
    source.replace(at, 3, dtype.str());
  return context.parse(source);
}
} // namespace

TEST(IndirectMaskCostModelTest, RankFactorPrefixAndExplicitRandomPrior) {
  IRTestContext context(true);
  const IndirectGatherMemoryCostModel model;
  for (bool store : {false, true})
    for (bool random : {false, true})
      for (int64_t rank = 1; rank <= 5; ++rank)
        for (int64_t factor : {1, 2, 4}) {
          const int64_t fi = factor == 4 ? 2 : factor == 2 ? 1 : 0;
          const int64_t e = 32 * (1LL << ((rank + 2 + fi) % 3));
          SmallVector<int64_t> shape(rank, 2);
          shape.back() = e / (1LL << (rank - 1));
          auto module = maskModule(context, shape, e / 2, store, random);
          ASSERT_TRUE(module);
          MemoryCase test(context, Float32Type::get(&context), shape, store);
          module->walk([&](Operation *op) {
            if (op->getName().getStringRef() ==
                (store ? "tt.store" : "tt.load"))
              test.stage.operations = {op};
          });
          auto profile = calibrationProfile();
          profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
              "prefix_mask_rank_factor_20261017";
          EXPECT_TRUE(profile.isValid());
          const auto prefix =
              model.cost(test.stage, profile, {StageMode::SIMT, factor});
          EXPECT_EQ(bool(store ? prefix.store : prefix.load), !random);
          profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
              "random_mask_rank_factor_20261017";
          const auto result =
              model.cost(test.stage, profile, {StageMode::SIMT, factor});
          ASSERT_TRUE(store ? result.store : result.load);
          EXPECT_GT(*(store ? result.store : result.load), 0);
          test.stage.iterationCount = 1;
          test.stage.legalSimtFactors = {factor};
          test.stage.workload.operationCounts.clear();
          test.stage.workload.issueElements = 0;
          for (const auto &cost : indirectCosts(test.stage, profile)) {
            if (cost.implementation.mode != StageMode::SIMT)
              continue;
            const auto price =
                model.cost(test.stage, profile, cost.implementation);
            ASSERT_TRUE(store ? price.store : price.load);
            const double memory = *(store ? price.store : price.load);
            EXPECT_NEAR(store ? cost.resources.store : cost.resources.load,
                        memory, 1e-10);
            EXPECT_NEAR(cost.totalCycles,
                        cost.resources.setup +
                            memory * cost.implementation.superblockFactor,
                        1e-8);
          }
          EXPECT_FALSE(
              model.cost(test.stage, profile, {StageMode::SIMT, 8}).load);
          EXPECT_FALSE(
              model.cost(test.stage, profile, {StageMode::SIMT, 8}).store);
          profile.baseSimtWarpCount = 16; // q<1 must return to #2528.
          const auto small =
              model.cost(test.stage, profile, {StageMode::SIMT, factor});
          EXPECT_FALSE(small.load || small.store);
        }
}

TEST(IndirectMaskCostModelTest, PrefixRejectsWrongCountAndReuse) {
  IRTestContext context(true);
  MemoryCase test(context, Float32Type::get(&context), {32}, true);
  auto profile = calibrationProfile();
  profile.simtIndirectStoreModel = "prefix_mask_rank_factor_20261017";
  const IndirectGatherMemoryCostModel model;
  for (int64_t k : {8, 16}) {
    auto module = maskModule(context, {32}, k, true);
    ASSERT_TRUE(module);
    module->walk([&](triton::StoreOp op) { test.stage.operations = {op}; });
    auto result = model.cost(test.stage, profile, {StageMode::SIMT, 1});
    EXPECT_EQ(bool(result.store), k == 16); // q=1 is included.
    if (result.store)
      EXPECT_NEAR(*result.store, 166.52232075147882, 1e-10);
    test.stage.workload.hasProvenIndirectStoreReuse = true;
    EXPECT_FALSE(model.cost(test.stage, profile, {StageMode::SIMT, 1}).store);
    test.stage.workload.hasProvenIndirectStoreReuse = false;
  }
}

TEST(IndirectMaskCostModelTest, PartitionedRankThreeUsesMaskFormula) {
  IRTestContext context(true);
  for (bool store : {false, true}) {
    auto module = maskModule(context, {2, 2, 32}, 64, store);
    ASSERT_TRUE(module);
    auto partition = StagePartitioner().partition(*module, SimtAnchorPlan{},
                                                  StagePartitionerOptions{});
    ASSERT_TRUE(bool(partition));
    auto profile = calibrationProfile();
    profile.simtIndirectLoadModel = profile.simtIndirectStoreModel =
        "prefix_mask_rank_factor_20261017";
    unsigned hits = 0;
    for (const auto &stage : partition->stages) {
      auto result = IndirectGatherMemoryCostModel().cost(stage, profile,
                                                         {StageMode::SIMT, 1});
      hits += bool(store ? result.store : result.load);
    }
    EXPECT_EQ(hits, 1u);
  }
}
