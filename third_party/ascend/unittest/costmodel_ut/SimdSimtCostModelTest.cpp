// Tests for SimdSimtCostModel responsibilities.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "CostModelTestUtils.h"
#include "StageIRTestUtils.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(SimdSimtCostModelTest, StageHasOnlySimdOrSimtImplementations) {
  LogicalStage stage = logicalStage("scalar", StageCostModelKind::ScalarIssue);
  auto table = evaluateOneStage(std::move(stage));
  ASSERT_TRUE(bool(table)) << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  EXPECT_EQ(table->stages.front().implementations[0].implementation.mode,
            StageMode::SIMD);
  EXPECT_EQ(table->stages.front().implementations[1].implementation.mode,
            StageMode::SIMT);
}

TEST(SimdSimtCostModelTest, GenericSemanticStagesDoNotRequireAWorkloadDomain) {
  IRTestContext context(false, true);
  auto module = context.parse(R"mlir(
    module {
      func.func @unrelated_kernel(%pointer: i64) {
        %zero = arith.constant dense<0.0> : tensor<16xf32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16xf32>
        %mask = arith.cmpf ogt, %loaded, %zero : tensor<16xf32>
        "tt.store"(%pointer, %loaded) : (i64, tensor<16xf32>) -> ()
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  ASSERT_TRUE(bool(partition)) << llvm::toString(partition.takeError());

  ASSERT_TRUE(partition->operationOwnershipComplete);
  EXPECT_EQ(partition->modeledOperationCount, 4);
  ASSERT_EQ(partition->stages.size(), 4u);
  EXPECT_EQ(partition->stages[0].costModelKind,
            StageCostModelKind::ScalarIssue);
  EXPECT_EQ(partition->stages[1].costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_EQ(partition->stages[2].costModelKind,
            StageCostModelKind::PredicateMask);
  EXPECT_EQ(partition->stages[3].costModelKind,
            StageCostModelKind::ContinuousTileStore);
}
