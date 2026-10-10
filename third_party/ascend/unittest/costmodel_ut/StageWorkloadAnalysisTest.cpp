// Tests for StageWorkloadAnalysis responsibilities.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "CostModelTestUtils.h"
#include "StageIRTestUtils.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

TEST(StageWorkloadAnalysisTest,
     WorkloadDoesNotInferSegmentsOrScalarFallbackFromLogicalRows) {
  IRTestContext context;
  auto module = context.parse(R"mlir(
    module {
      func.func @kernel(%a4x16: tensor<4x16xf32>,
                        %b4x16: tensor<4x16xf32>,
                        %a8x8: tensor<8x8xf32>,
                        %b8x8: tensor<8x8xf32>,
                        %a16x4: tensor<16x4xf32>,
                        %b16x4: tensor<16x4xf32>) {
        %four_by_sixteen = arith.addf %a4x16, %b4x16 : tensor<4x16xf32>
        %eight_by_eight = arith.addf %a8x8, %b8x8 : tensor<8x8xf32>
        %sixteen_by_four = arith.addf %a16x4, %b16x4 : tensor<16x4xf32>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("short_axis_geometry", StageCostModelKind::ScalarMath);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &workloads =
      partition.stages.front().workload.tensorOperationWorkloads;
  ASSERT_EQ(workloads.size(), 1u);
  EXPECT_EQ(workloads.front().contiguousElementsPerSegment, 64);
  EXPECT_EQ(workloads.front().elementBitWidth, 32);
  EXPECT_DOUBLE_EQ(workloads.front().logicalElements, 192.0);
  EXPECT_DOUBLE_EQ(workloads.front().segmentCount, 3.0);
}

TEST(StageWorkloadAnalysisTest,
     WorkloadRetainsRowBroadcastAndComparisonInputWidth) {
  IRTestContext context(false, true);
  auto module = context.parse(R"mlir(
    module {
      func.func @kernel(%a: tensor<8x8xf32>, %row: tensor<8x1xf32>,
                        %a3: tensor<2x4x8xf32>, %row3: tensor<2x4x1xf32>) {
        %b = "tt.broadcast"(%row) : (tensor<8x1xf32>) -> tensor<8x8xf32>
        %sum = arith.addf %a, %b : tensor<8x8xf32>
        %mask = arith.cmpf olt, %a, %sum : tensor<8x8xf32>
        %b3 = "tt.broadcast"(%row3) : (tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
        %sum3 = arith.addf %a3, %b3 : tensor<2x4x8xf32>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("broadcast", StageCostModelKind::ScalarMath);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    if (mlir::isa<mlir::arith::AddFOp, mlir::arith::CmpFOp>(operation))
      stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &workloads =
      partition.stages.front().workload.tensorOperationWorkloads;
  ASSERT_EQ(workloads.size(), 2u);
  for (const auto &workload : workloads) {
    EXPECT_EQ(workload.elementBitWidth, 32);
    if (workload.operation == "f32.add") {
      EXPECT_EQ(workload.contiguousElementsPerSegment, 8);
      EXPECT_DOUBLE_EQ(workload.segmentCount, 16);
    } else {
      EXPECT_EQ(workload.operation, "predicate.cmp");
      EXPECT_EQ(workload.contiguousElementsPerSegment, 64);
      EXPECT_DOUBLE_EQ(workload.segmentCount, 1);
    }
  }
}

TEST(StageWorkloadAnalysisTest, WorkloadUsesPostFlattenBroadcastSuffix) {
  IRTestContext context(false, true);
  auto module = context.parse(R"mlir(
    module {
      func.func @kernel(%x: tensor<2x4x8xf32>,
                        %leading: tensor<1x4x8xf32>,
                        %middle: tensor<2x1x8xf32>,
                        %x4: tensor<2x4x8x4xf32>,
                        %leading4: tensor<1x4x8x4xf32>,
                        %unit: tensor<2x4x8x1xf32>,
                        %row: tensor<2x4x1x1xf32>,
                        %ix: tensor<2x4x8xi32>, %iy: tensor<2x1x8xi32>) {
        %a = "tt.broadcast"(%leading) : (tensor<1x4x8xf32>) -> tensor<2x4x8xf32>
        %s0 = arith.addf %x, %a : tensor<2x4x8xf32>
        %b = "tt.broadcast"(%middle) : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
        %s1 = arith.addf %x, %b : tensor<2x4x8xf32>
        %c = "tt.broadcast"(%leading4) : (tensor<1x4x8x4xf32>) -> tensor<2x4x8x4xf32>
        %s2 = arith.addf %x4, %c : tensor<2x4x8x4xf32>
        %d = "tt.broadcast"(%row) : (tensor<2x4x1x1xf32>) -> tensor<2x4x8x1xf32>
        %s3 = arith.addf %unit, %d : tensor<2x4x8x1xf32>
        %e = "tt.broadcast"(%iy) : (tensor<2x1x8xi32>) -> tensor<2x4x8xi32>
        %s4 = arith.addi %ix, %e : tensor<2x4x8xi32>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator()) {
    if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(operation))
      continue;
    LogicalStage stage =
        logicalStage("flatten", StageCostModelKind::ScalarMath);
    stage.operations.push_back(&operation);
    partition.stages.push_back(std::move(stage));
  }
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  ASSERT_EQ(partition.stages.size(), 5u);
  const int64_t widths[] = {32, 8, 128, 8, 8};
  const double segments[] = {2, 8, 2, 8, 8};
  for (size_t i = 0; i < partition.stages.size(); ++i) {
    const auto &groups = partition.stages[i].workload.tensorOperationWorkloads;
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].contiguousElementsPerSegment, widths[i]);
    EXPECT_DOUBLE_EQ(groups[0].segmentCount, segments[i]);
  }
}

TEST(StageWorkloadAnalysisTest,
     WorkloadPreservesAtomicSemanticsWithoutCountingAStore) {
  IRTestContext context(false, true);
  auto module = context.parse(R"mlir(
    module {
      func.func @kernel(%load_pointer: tensor<64x!tt.ptr<f32>>,
                        %index_pointer: tensor<64x!tt.ptr<i64>>,
                        %base_pointer: tensor<64x!tt.ptr<f32>>,
                        %value: tensor<64xf32>) {
        %regular = "tt.load"(%load_pointer)
          : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        %index = "tt.load"(%index_pointer)
          : (tensor<64x!tt.ptr<i64>>) -> tensor<64xi64>
        %address = "tt.addptr"(%base_pointer, %index)
          : (tensor<64x!tt.ptr<f32>>, tensor<64xi64>)
            -> tensor<64x!tt.ptr<f32>>
        %mask = arith.constant dense<true> : tensor<64xi1>
        %old = "tt.atomic_rmw"(%address, %value, %mask)
          {atomic_rmw_op = 5 : i32, sem = 4 : i32, scope = 1 : i32}
          : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>, tensor<64xi1>)
            -> tensor<64xf32>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("atomic_workload", StageCostModelKind::AtomicMemory);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  mlir::Block &body = function.getBody().front();
  for (mlir::Operation &operation : body.without_terminator())
    stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));

  const LogicalStage &analyzed = partition.stages.front();
  EXPECT_TRUE(analyzed.features.hasAtomicMemory);
  EXPECT_TRUE(analyzed.features.hasIndirectMemory);
  EXPECT_TRUE(analyzed.features.hasContiguousMemory);
  EXPECT_DOUBLE_EQ(analyzed.workload.storeBytes, 0.0);
  EXPECT_DOUBLE_EQ(analyzed.workload.storeWarpInstructions, 0.0);
  ASSERT_EQ(analyzed.workload.atomicWorkloads.size(), 1u);
  const auto &atomic = analyzed.workload.atomicWorkloads.front();
  EXPECT_EQ(atomic.kind, "fadd");
  EXPECT_EQ(atomic.dataType, "f32");
  EXPECT_EQ(atomic.memorySemantic, "acq_rel");
  EXPECT_EQ(atomic.memoryScope, "gpu");
  EXPECT_DOUBLE_EQ(atomic.logicalElements, 64.0);
  EXPECT_FALSE(atomic.resultUsed);
  EXPECT_TRUE(atomic.addressDependsOnLoadedIndex);
  EXPECT_TRUE(atomic.contentionUnknown);
  EXPECT_EQ(analyzed.costModelKind, StageCostModelKind::AtomicMemory);
}
