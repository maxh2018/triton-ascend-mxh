// Tests for StagePartitioner responsibilities.
#include "AscendModel/Analysis/StagePartitioner.h"
#include "CostModelTestUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir::ascend;
using namespace mlir::ascend::test;

namespace {
SimdSimtFeatureSummary triangularBt16StageFeatures() {
  SimdSimtFeatureSummary f;
  f.simtAnchors.count = 1;
  TriangularSolveFacts triangular;
  triangular.blockRows = 16;
  triangular.blockColumns = 16;
  triangular.accumulatorType = "f32";
  triangular.recurrenceStartRow = 2;
  triangular.recurrenceLoopCount = 14;
  f.simtAnchors.triangularSolves.push_back(triangular);
  return f;
}
} // namespace

TEST(StagePartitionerTest,
     ScopeSuperBlockLegalityRequiresBackendAndResourceMaximum) {
  auto makePartition = []() {
    StagePartition partition;
    LogicalStage stage =
        logicalStage("payload", StageCostModelKind::LoopCarriedRecurrence,
                     StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.localSimtMaterializable = true;
    stage.localSuperblockMaterializable = true;
    stage.localSimtFactors = {1};
    partition.stages.push_back(std::move(stage));
    return partition;
  };

  StagePartition f1Only = makePartition();
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(f1Only, 4, false))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(f1Only.stages[0].localSimtFactors, (std::vector<int64_t>{1}));

  StagePartition scopeSuperblock = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(scopeSuperblock, 4, true, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(scopeSuperblock.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));

  // ABI-v2 creates an F1 V1 scheduling loop and refines only the selected
  // scope after bufferization, but local and whole-kernel factors still share
  // the same target/runtime warp-resource maximum.
  StagePartition mixedOnly = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(mixedOnly, 1, true, 1))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(mixedOnly.stages[0].legalSimtFactors, (std::vector<int64_t>{1}));
  EXPECT_EQ(mixedOnly.stages[0].localSimtFactors, (std::vector<int64_t>{1}));
  auto mixedOnlyCosts = evaluateOneStage(mixedOnly.stages[0]);
  if (!mixedOnlyCosts)
    FAIL() << llvm::toString(mixedOnlyCosts.takeError());
  ASSERT_EQ(mixedOnlyCosts->stages[0].implementations.size(), 3u);
  EXPECT_EQ(mixedOnlyCosts->stages[0].legalSimtFactors,
            (std::vector<int64_t>{1}));
  EXPECT_EQ(mixedOnlyCosts->stages[0].localSimtFactors,
            (std::vector<int64_t>{1}));
}

TEST(StagePartitionerTest, LocalScopeFactorsHonorKernelResourceMaximum) {
  StagePartition partition;
  LogicalStage stage;
  stage.id = "indirect_tile_gather";
  stage.costModelKind = StageCostModelKind::IndirectGatherMemory;
  stage.scheduleKind = StageScheduleKind::PartiallyDependent;
  stage.iterationCount = 1;
  stage.localSimtMaterializable = true;
  stage.localSuperblockMaterializable = true;
  partition.stages.push_back(std::move(stage));

  ASSERT_FALSE(StageModeLegalityAnalysis().analyze(
      partition, /*maximumSuperblockFactor=*/2,
      /*scopeSuperblockMaterializable=*/true,
      /*maximumScopeSuperblockFactor=*/2));
  const LogicalStage &result = partition.stages.front();
  EXPECT_EQ(result.legalSimtFactors, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(result.localSimtFactors, (std::vector<int64_t>{1, 2}));
}

TEST(StagePartitionerTest,
     OperationGraphBoundaryOwnsEveryRootAndDerivesLiveValues) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::scf::ForOp loop) { recurrence = loop; });
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations.push_back(recurrence);
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.triangularSolve =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto structure =
      mlir::ascend::ProgramStructureAnalysis().analyze(*module, anchorPlan);
  if (!structure)
    FAIL() << llvm::toString(structure.takeError());
  EXPECT_EQ(structure->rootOperations.size(), 6u);

  auto result = StagePartitioner().partition(*module, anchorPlan,
                                             StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  const StagePartition &partition = *result;
  EXPECT_TRUE(partition.operationOwnershipComplete);

  int64_t ownedRootCount = 0;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalStage &stage : partition.stages) {
    ownedRootCount += static_cast<int64_t>(stage.operations.size());
    if (llvm::is_contained(stage.operations, recurrence))
      recurrenceStage = &stage;
  }
  EXPECT_EQ(ownedRootCount, partition.modeledOperationCount);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(recurrenceStage->operations.size(), 1u);
  EXPECT_FALSE(recurrenceStage->liveIns.empty());
  EXPECT_EQ(recurrenceStage->liveOuts.size(), 1u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
  EXPECT_TRUE(recurrenceStage->localSimtMaterializable);
  // The arith.addf is the scf.for body and therefore represents 256 element
  // additions on every one of the 14 dynamic recurrence iterations.  The
  // per-iteration Stage workload must remain 256, not be divided by 14.
  auto add = recurrenceStage->workload.operationElements.find("f32.add");
  ASSERT_NE(add, recurrenceStage->workload.operationElements.end());
  EXPECT_DOUBLE_EQ(add->second, 256.0);
}

TEST(StagePartitionerTest,
     SameStatementSupportOperationsJoinTheDominantResourceStage) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %index = arith.constant 0 : i64
        %mask = arith.cmpi eq, %index, %index : i64
        %value = "tt.load"(%pointer, %mask) : (i64, i1) -> f32
        %tail = arith.constant 1 : i64
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *> roots;
  module->walk([&](mlir::func::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        roots.push_back(&operation);
  });
  ASSERT_EQ(roots.size(), 4u);
  auto statement = mlir::FileLineColLoc::get(&context, "kernel.py", 10, 1);
  for (mlir::Operation *operation : llvm::ArrayRef(roots).take_front(3))
    operation->setLoc(statement);
  roots.back()->setLoc(mlir::FileLineColLoc::get(&context, "kernel.py", 11, 1));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations.assign(roots.begin(), roots.end());
  auto result = mlir::ascend::StageBoundaryAnalysis().analyze(
      structure, mlir::ascend::SimtAnchorPlan{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->stages.size(), 2u);
  EXPECT_EQ(result->stages.front().operations.size(), 3u);
  EXPECT_EQ(result->stages.front().costModelKind,
            StageCostModelKind::ScalarLoad);
  ASSERT_EQ(result->stages.back().operations.size(), 1u);
  EXPECT_EQ(result->stages.back().operations.front(), roots.back());
}

TEST(StagePartitionerTest,
     CompoundScopeOrderIsNormalizedBeforeStagePartitioning) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %setup = arith.constant dense<0> : tensor<16xi32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *setup = nullptr;
  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "arith.constant" &&
        operation->getNumResults() == 1 &&
        mlir::isa<mlir::RankedTensorType>(operation->getResult(0).getType()))
      setup = operation;
    if (mlir::isa<mlir::scf::ForOp>(operation))
      recurrence = operation;
  });
  ASSERT_NE(setup, nullptr);
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations = {setup, recurrence};
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.triangularSolve =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto structure =
      mlir::ascend::ProgramStructureAnalysis().analyze(*module, anchorPlan);
  if (!structure)
    FAIL() << llvm::toString(structure.takeError());
  auto setupPosition = llvm::find(structure->rootOperations, setup);
  auto recurrencePosition = llvm::find(structure->rootOperations, recurrence);
  ASSERT_NE(setupPosition, structure->rootOperations.end());
  ASSERT_NE(recurrencePosition, structure->rootOperations.end());
  EXPECT_EQ(recurrencePosition - setupPosition, 1);

  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  const LogicalStage *loadStage = nullptr;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalStage &stage : partition->stages) {
    if (llvm::any_of(stage.operations, [&](mlir::Operation *operation) {
          return operation->getName().getStringRef() == "tt.load";
        }))
      loadStage = &stage;
    if (llvm::is_contained(stage.operations, recurrence))
      recurrenceStage = &stage;
  }
  ASSERT_NE(loadStage, nullptr);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(loadStage->operations.size(), 1u);
  EXPECT_EQ(recurrenceStage->operations.size(), 2u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
}

TEST(StagePartitionerTest,
     NestedLocalScopeDoesNotAdvertiseUnsupportedSuperBlockFactors) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64, %condition: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          scf.if %condition {
            %value = "tt.load"(%pointer) : (i64) -> tensor<16xf32>
          }
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::scf::ForOp v1Loop;
  mlir::Operation *nestedLoad = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
      v1Loop = loop;
    if (operation->getName().getStringRef() == "tt.load")
      nestedLoad = operation;
  });
  ASSERT_TRUE(v1Loop);
  ASSERT_NE(nestedLoad, nullptr);
  v1Loop->setAttr("ta.auto_blockify_v1.loop", mlir::UnitAttr::get(&context));

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = nestedLoad;
  anchor.scopeOperations.push_back(nestedLoad);
  anchor.scopeInsertionPoint = nestedLoad;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  StagePartitionerOptions options;
  options.maximumSuperblockFactor = 4;
  options.scopeSuperblockMaterializable = true;
  auto result = StagePartitioner().partition(*module, anchorPlan, options);
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *nestedStage = nullptr;
  for (const LogicalStage &stage : result->stages)
    if (!stage.simtAnchorIndices.empty())
      nestedStage = &stage;
  ASSERT_NE(nestedStage, nullptr);
  EXPECT_TRUE(nestedStage->localSimtMaterializable);
  EXPECT_FALSE(nestedStage->localSuperblockMaterializable);
  EXPECT_FALSE(nestedStage->operations.empty());
  EXPECT_EQ(nestedStage->localSimtFactors, (std::vector<int64_t>{1}));

  auto costs = StageCostEvaluator().evaluate(*result, hardwareProfile());
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  auto nestedCost = llvm::find_if(costs->stages, [&](const auto &stage) {
    return stage.id == nestedStage->id;
  });
  ASSERT_NE(nestedCost, costs->stages.end());
  EXPECT_FALSE(nestedCost->sourceLocations.empty());
}

TEST(StagePartitionerTest, ClippedDynamicLoopUsesStaticTripCountCap) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  for (bool capped : {false, true}) {
    std::string source = R"mlir(
      module {
        func.func @kernel(%limit: i32, %initial: i32) -> i32 {
          %c1 = arith.constant 1 : i32
          %c2 = arith.constant 2 : i32
          %c16 = arith.constant 16 : i32
          %cap = arith.minsi %limit, %c16 : i32
          %result = scf.for %i = %c2 to UPPER step %c1
              iter_args(%state = %initial) -> (i32) : i32 {
            %next = arith.addi %state, %i : i32
            scf.yield %next : i32
          }
          %second = scf.for %j = %c2 to %cap step %c1
              iter_args(%state = %result) -> (i32) : i32 {
            %next = arith.addi %state, %j : i32
            scf.yield %next : i32
          }
          return %second : i32
        }
      }
    )mlir";
    source.replace(source.find("UPPER"), 5, capped ? "%cap" : "%limit");
    if (!capped)
      source.replace(source.find("to %cap"), 7, "to %limit");
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    mlir::ascend::SimtAnchorPlan anchors;
    auto partition = StagePartitioner().partition(*module, anchors,
                                                  StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    if (llvm::Error error = StageFeatureAnalysis().analyze(*partition))
      FAIL() << llvm::toString(std::move(error));
    int64_t maximumIterations = 1;
    for (const auto &stage : partition->stages) {
      maximumIterations = std::max(maximumIterations, stage.iterationCount);
      EXPECT_EQ(stage.features.parallelRecurrenceGroupCount, 1);
    }
    EXPECT_EQ(maximumIterations, capped ? 14 : 1);
  }
}

TEST(StagePartitionerTest, AdjacentStructuredLoopsRemainSerialStages) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @two_serial_loops(%pointer: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %first = "tt.load"(%pointer) : (i64) -> f32
        }
        scf.for %i = %c0 to %c4 step %c1 {
          %second = "tt.load"(%pointer) : (i64) -> f32
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  llvm::SmallVector<const LogicalStage *> loopStages;
  for (const LogicalStage &stage : partition->stages)
    if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
      loopStages.push_back(&stage);
  ASSERT_EQ(loopStages.size(), 2u);
  EXPECT_NE(loopStages[0]->operations.front(),
            loopStages[1]->operations.front());
}

TEST(StagePartitionerTest,
     LocalScopeReturningPointerTensorIsRejectedBeforeScoring) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%base: !tt.ptr<f16>) {
        %indices = arith.constant dense<0> : tensor<16xi32>
        %ptrs = "tt.addptr"(%base, %indices)
            : (!tt.ptr<f16>, tensor<16xi32>) -> tensor<16x!tt.ptr<f16>>
        %values = "tt.load"(%ptrs)
            : (tensor<16x!tt.ptr<f16>>) -> tensor<16xf16>
        %reduced = "tt.reduce"(%values) : (tensor<16xf16>) -> f16
        "tt.store"(%base, %reduced) : (!tt.ptr<f16>, f16) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *> roots;
  module->walk([&](mlir::func::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        roots.push_back(&operation);
  });
  ASSERT_EQ(roots.size(), 5u);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = roots[1];
  anchor.scopeOperations = {roots[1]};
  anchor.scopeInsertionPoint = roots[1];
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations.assign(roots.begin(), roots.end());
  auto result =
      mlir::ascend::StageBoundaryAnalysis().analyze(structure, anchorPlan);
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *gather = nullptr;
  for (const LogicalStage &stage : result->stages)
    if (llvm::is_contained(stage.operations, roots[1]))
      gather = &stage;
  ASSERT_NE(gather, nullptr);
  EXPECT_FALSE(gather->localSimtMaterializable);
  EXPECT_TRUE(gather->localSimtFactors.empty());
  EXPECT_TRUE(gather->simtAnchorIndices.empty());
}

TEST(StagePartitionerTest, PointerInductionLoopIsNotADataRecurrence) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%start: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %step = arith.constant 16 : i64
        %address = scf.for %i = %c0 to %c8 step %c1
            iter_args(%current = %start) -> i64 {
          %value = "tt.load"(%current) : (i64) -> f32
          %next = arith.addi %current, %step : i64
          scf.yield %next : i64
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *loop = nullptr;
  module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  ASSERT_NE(loop, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("pointer_loop", StageCostModelKind::ConversionPack,
                   StageScheduleKind::IndependentPipelined, 8);
  stage.operations.push_back(loop);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));
  const LogicalStage &classified = partition.stages.front();
  EXPECT_TRUE(classified.features.hasLoop);
  EXPECT_TRUE(classified.features.hasPointerInduction);
  EXPECT_FALSE(classified.features.hasLoopCarriedDataDependency);
  EXPECT_EQ(classified.costModelKind,
            StageCostModelKind::IndependentPipelinedLoop);
}

TEST(StagePartitionerTest, IncompatibleDominantStructuresRequireStageSplit) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("gather_dot", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::PartiallyDependent, 1);
  stage.features.hasDot = true;
  stage.features.hasIndirectMemory = true;
  partition.stages.push_back(std::move(stage));

  llvm::Error error =
      mlir::ascend::StageKindClassifier().analyze(partition, 16384);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("requires_split"),
            std::string::npos);
}
