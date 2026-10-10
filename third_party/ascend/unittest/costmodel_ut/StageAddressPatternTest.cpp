// Tests for StageAddressPattern responsibilities.
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

static std::string partialContinuousTileIR(int64_t rows, int64_t columns,
                                           int64_t columnStep,
                                           bool storePayload = false) {
  std::string source = R"mlir(
    module {
      func.func @kernel(%indices: !tt.ptr<i32>,
                        %base: !tt.ptr<f32>,
                        %data: tensor<@ROWS@x@COLS@xf32>) {
        %indexRange = "tt.make_range"() {start = 0 : i32, end = @ROWS@ : i32}
          : () -> tensor<@ROWS@xi32>
        %indexBase = "tt.splat"(%indices)
          : (!tt.ptr<i32>) -> tensor<@ROWS@x!tt.ptr<i32>>
        %indexPointer = "tt.addptr"(%indexBase, %indexRange)
          : (tensor<@ROWS@x!tt.ptr<i32>>, tensor<@ROWS@xi32>)
            -> tensor<@ROWS@x!tt.ptr<i32>>
        %index = tt.load %indexPointer : tensor<@ROWS@x!tt.ptr<i32>>
        %index2d = "tt.expand_dims"(%index) {axis = 1 : i32}
          : (tensor<@ROWS@xi32>) -> tensor<@ROWS@x1xi32>
        %rowStride = arith.constant dense<@ROWSTRIDE@>
          : tensor<@ROWS@x1xi32>
        %rowOffset = arith.muli %index2d, %rowStride
          : tensor<@ROWS@x1xi32>
        @ROW_BROADCAST@
        %columns = "tt.make_range"() {start = 0 : i32, end = @COLS@ : i32}
          : () -> tensor<@COLS@xi32>
        %column2d = "tt.expand_dims"(%columns) {axis = 0 : i32}
          : (tensor<@COLS@xi32>) -> tensor<1x@COLS@xi32>
        %columnBroadcast = "tt.broadcast"(%column2d)
          : (tensor<1x@COLS@xi32>) -> tensor<@ROWS@x@COLS@xi32>
        %columnStride = arith.constant dense<@STEP@>
          : tensor<@ROWS@x@COLS@xi32>
        %columnOffset = arith.muli %columnBroadcast, %columnStride
          : tensor<@ROWS@x@COLS@xi32>
        %offset = arith.addi @ROW_VALUE@, %columnOffset
          : tensor<@ROWS@x@COLS@xi32>
        %baseBroadcast = "tt.splat"(%base)
          : (!tt.ptr<f32>) -> tensor<@ROWS@x@COLS@x!tt.ptr<f32>>
        %pointer = "tt.addptr"(%baseBroadcast, %offset)
          : (tensor<@ROWS@x@COLS@x!tt.ptr<f32>>,
             tensor<@ROWS@x@COLS@xi32>)
            -> tensor<@ROWS@x@COLS@x!tt.ptr<f32>>
        @PAYLOAD@
        return
      }
    }
  )mlir";
  auto replace = [&](llvm::StringRef from, llvm::StringRef to) {
    size_t position = 0;
    while ((position = source.find(from.str(), position)) !=
           std::string::npos) {
      source.replace(position, from.size(), to.str());
      position += to.size();
    }
  };
  replace("@ROW_BROADCAST@",
          columns == 1 ? ""
                       : "%rowBroadcast = \"tt.broadcast\"(%rowOffset)\n"
                         "          : (tensor<@ROWS@x1xi32>) -> "
                         "tensor<@ROWS@x@COLS@xi32>");
  replace("@ROW_VALUE@", columns == 1 ? "%rowOffset" : "%rowBroadcast");
  replace("@ROWS@", std::to_string(rows));
  replace("@COLS@", std::to_string(columns));
  replace("@ROWSTRIDE@", std::to_string(columns * columnStep));
  replace("@STEP@", std::to_string(columnStep));
  replace("@PAYLOAD@",
          storePayload
              ? "tt.store %pointer, %data : tensor<" + std::to_string(rows) +
                    "x" + std::to_string(columns) + "x!tt.ptr<f32>>"
              : "%payload = tt.load %pointer : tensor<" + std::to_string(rows) +
                    "x" + std::to_string(columns) + "x!tt.ptr<f32>>");
  return source;
}

} // namespace

TEST(StageAddressPatternTest, PartialContinuousRowsUseDirectMemoryPerRow) {
  for (int64_t columns : {32, 8, 1}) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::triton::TritonDialect>();
    context.allowUnregisteredDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        partialContinuousTileIR(8, columns, 1), &context);
    ASSERT_TRUE(module);
    auto partition = StagePartitioner().partition(
        *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    const LogicalStage *payload = nullptr;
    for (const LogicalStage &stage : partition->stages)
      if (stage.costModelKind ==
          StageCostModelKind::PartialContinuousTileMemory)
        payload = &stage;
    ASSERT_NE(payload, nullptr) << "columns=" << columns;
    ASSERT_EQ(payload->workload.addressPatterns.size(), 1u);
    const auto &address = payload->workload.addressPatterns.front();
    EXPECT_EQ(address.stageId, payload->id);
    EXPECT_TRUE(address.dependsOnLoadedValue);
    ASSERT_EQ(address.axes.size(), 2u);
    EXPECT_EQ(address.axes[0].extent, 8);
    EXPECT_EQ(address.axes[0].regularity, "opaque_loaded");
    if (columns > 1) {
      EXPECT_EQ(address.axes[1].regularity, "fixed_stride");
      ASSERT_TRUE(address.axes[1].knownStride.has_value());
      EXPECT_EQ(*address.axes[1].knownStride, 1);
    }
    EXPECT_TRUE(payload->features.hasPartialContinuousMemory);
    EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadRows, 8.0);
    EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadBytes,
                     8.0 * columns * 4.0);
    EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadWarpInstructions,
                     8.0 * std::ceil(columns / 32.0));
    EXPECT_DOUBLE_EQ(payload->workload.indirectLoadTransactions, 0.0);
    auto table = evaluateOneStage(*payload);
    if (!table)
      FAIL() << llvm::toString(table.takeError());
    ASSERT_GE(table->stages.front().implementations.size(), 2u);
    EXPECT_DOUBLE_EQ(table->stages.front().implementations[0].resources.load,
                     8.0 * columns * 4.0 / 32.0);
  }
}

TEST(StageAddressPatternTest, PartialStructuredStoreAndNonunitStride) {
  for (bool store : {false, true}) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::triton::TritonDialect>();
    context.allowUnregisteredDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        partialContinuousTileIR(8, 8, 2, store), &context);
    ASSERT_TRUE(module);
    auto partition = StagePartitioner().partition(
        *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    bool sawExpectedKind = false;
    for (const LogicalStage &stage : partition->stages) {
      if (store && stage.costModelKind ==
                       StageCostModelKind::PartialContinuousTileMemory) {
        sawExpectedKind = true;
        EXPECT_DOUBLE_EQ(stage.workload.partialContinuousStoreRows, 8.0);
        EXPECT_DOUBLE_EQ(stage.workload.partialContinuousStoreWarpInstructions,
                         8.0);
        EXPECT_DOUBLE_EQ(stage.workload.indirectStoreTransactions, 0.0);
      }
      if (!store && stage.costModelKind ==
                        StageCostModelKind::PartialContinuousTileMemory) {
        sawExpectedKind = true;
        EXPECT_DOUBLE_EQ(stage.workload.partialContinuousLoadRows, 8.0);
        EXPECT_DOUBLE_EQ(stage.workload.indirectLoadTransactions, 0.0);
      }
    }
    EXPECT_TRUE(sawExpectedKind);
  }

  // OffsetAnalysis cannot parse a tensor-pointer function argument as a
  // pointer root.  Keep this unsupported form in the indirect class.
  std::string unsupported = partialContinuousTileIR(8, 8, 1);
  size_t argument = unsupported.find("%indices: !tt.ptr<i32>");
  ASSERT_NE(argument, std::string::npos);
  unsupported.replace(argument, std::string("%indices: !tt.ptr<i32>").size(),
                      "%indices: tensor<8x!tt.ptr<i32>>");
  size_t begin = unsupported.find("        %indexRange =");
  size_t end = unsupported.find("        %index2d =", begin);
  ASSERT_NE(begin, std::string::npos);
  ASSERT_NE(end, std::string::npos);
  unsupported.replace(begin, end - begin,
                      "        %index = tt.load %indices : "
                      "tensor<8x!tt.ptr<i32>>\n");
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(unsupported, &context);
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(
      *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  for (const LogicalStage &stage : partition->stages)
    EXPECT_NE(stage.costModelKind,
              StageCostModelKind::PartialContinuousTileMemory);
}

TEST(StageAddressPatternTest,
     PartialUnstructuredPrefixDoesNotRequireLoadedIndex) {
  for (bool store : {false, true}) {
    std::string source = partialContinuousTileIR(8, 8, 2, store);
    size_t begin = source.find("        %index = tt.load");
    size_t end = source.find("        %index2d =", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    source.replace(begin, end - begin,
                   "        %index = arith.muli %indexRange, %indexRange : "
                   "tensor<8xi32>\n");
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::triton::TritonDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto partition = StagePartitioner().partition(
        *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    bool sawPayload = false;
    for (const LogicalStage &stage : partition->stages) {
      const double rows = store ? stage.workload.partialContinuousStoreRows
                                : stage.workload.partialContinuousLoadRows;
      if (rows == 0.0)
        continue;
      sawPayload = true;
      EXPECT_EQ(stage.costModelKind,
                StageCostModelKind::PartialContinuousTileMemory);
      EXPECT_DOUBLE_EQ(rows, 8.0);
    }
    EXPECT_TRUE(sawPayload);
  }
}

TEST(StageAddressPatternTest,
     PartialStructuredTailRequiresAllPrefixAxesUnstructured) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  std::string source = R"mlir(
module {
tt.func public @partial_3d(%src: !tt.ptr<f32>,
                            %indices: !tt.ptr<i32>) {
  %rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
  %indices_base = tt.splat %indices : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
  %indices_ptr = tt.addptr %indices_base, %rows
    : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
  %index = tt.load %indices_ptr : tensor<2x!tt.ptr<i32>>
  %index_2d = tt.expand_dims %index {axis = 1 : i32}
    : tensor<2xi32> -> tensor<2x1xi32>
  %index_3d = tt.expand_dims %index_2d {axis = 2 : i32}
    : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %index_broadcast = tt.broadcast %index_3d
    : tensor<2x1x1xi32> -> tensor<2x4x8xi32>
  %row_stride = arith.constant dense<32> : tensor<2x4x8xi32>
  %row_offset = arith.muli %index_broadcast, %row_stride
    : tensor<2x4x8xi32>

  %middle = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %middle_2d = tt.expand_dims %middle {axis = 0 : i32}
    : tensor<4xi32> -> tensor<1x4xi32>
  %middle_3d = tt.expand_dims %middle_2d {axis = 2 : i32}
    : tensor<1x4xi32> -> tensor<1x4x1xi32>
  %middle_broadcast = tt.broadcast %middle_3d
    : tensor<1x4x1xi32> -> tensor<2x4x8xi32>
  %middle_stride = arith.constant dense<8> : tensor<2x4x8xi32>
  %middle_offset = arith.muli %middle_broadcast, %middle_stride
    : tensor<2x4x8xi32>

  %columns = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %columns_2d = tt.expand_dims %columns {axis = 0 : i32}
    : tensor<8xi32> -> tensor<1x8xi32>
  %columns_3d = tt.expand_dims %columns_2d {axis = 0 : i32}
    : tensor<1x8xi32> -> tensor<1x1x8xi32>
  %column_offset = tt.broadcast %columns_3d
    : tensor<1x1x8xi32> -> tensor<2x4x8xi32>

  %row_and_middle = arith.addi %row_offset, %middle_offset
    : tensor<2x4x8xi32>
  %offset = arith.addi %row_and_middle, %column_offset
    : tensor<2x4x8xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x4x8x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset
    : tensor<2x4x8x!tt.ptr<f32>>, tensor<2x4x8xi32>
  %payload = tt.load %src_ptr : tensor<2x4x8x!tt.ptr<f32>>
  tt.return
}
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  auto partition = StagePartitioner().partition(
      *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  auto findPayload =
      [](const StagePartition &partition) -> const LogicalStage * {
    for (const LogicalStage &stage : partition.stages)
      for (mlir::Operation *operation : stage.operations) {
        if (operation->getName().getStringRef() != "tt.load" ||
            operation->getNumResults() != 1)
          continue;
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(
            operation->getResult(0).getType());
        if (type && type.getShape() == llvm::ArrayRef<int64_t>({2, 4, 8}))
          return &stage;
      }
    return nullptr;
  };
  const LogicalStage *payload = findPayload(*partition);
  ASSERT_NE(payload, nullptr);
  // [U, S, S]: a second structured suffix axis excludes the partial kind.
  EXPECT_EQ(payload->costModelKind, StageCostModelKind::IndirectGatherMemory);
  EXPECT_FALSE(payload->features.hasPartialContinuousMemory);
  EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadRows, 0.0);
  EXPECT_DOUBLE_EQ(payload->workload.indirectLoadBytes, 256.0);

  // [S, U, S]: a structured outer axis also excludes the partial kind.
  auto replaceOnce = [&](llvm::StringRef oldText, llvm::StringRef newText) {
    size_t position = source.find(oldText.str());
    ASSERT_NE(position, std::string::npos);
    source.replace(position, oldText.size(), newText.str());
  };
  replaceOnce("  %row_stride =",
              R"mlir(  %outer_2d = tt.expand_dims %rows {axis = 1 : i32}
    : tensor<2xi32> -> tensor<2x1xi32>
  %outer_3d = tt.expand_dims %outer_2d {axis = 2 : i32}
    : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %outer_broadcast = tt.broadcast %outer_3d
    : tensor<2x1x1xi32> -> tensor<2x4x8xi32>
  %row_stride =)mlir");
  replaceOnce("arith.muli %index_broadcast, %row_stride",
              "arith.muli %outer_broadcast, %row_stride");
  replaceOnce("  %middle_stride =",
              R"mlir(  %middle_idx_base = tt.splat %indices
    : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %middle_idx_ptr = tt.addptr %middle_idx_base, %middle
    : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %middle_idx = tt.load %middle_idx_ptr : tensor<4x!tt.ptr<i32>>
  %middle_idx_2d = tt.expand_dims %middle_idx {axis = 0 : i32}
    : tensor<4xi32> -> tensor<1x4xi32>
  %middle_idx_3d = tt.expand_dims %middle_idx_2d {axis = 2 : i32}
    : tensor<1x4xi32> -> tensor<1x4x1xi32>
  %middle_idx_broadcast = tt.broadcast %middle_idx_3d
    : tensor<1x4x1xi32> -> tensor<2x4x8xi32>
  %middle_stride =)mlir");
  replaceOnce("arith.muli %middle_broadcast, %middle_stride",
              "arith.muli %middle_idx_broadcast, %middle_stride");
  auto crossedModule =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  ASSERT_TRUE(crossedModule);
  auto crossed = StagePartitioner().partition(*crossedModule,
                                              mlir::ascend::SimtAnchorPlan{},
                                              StagePartitionerOptions{});
  if (!crossed)
    FAIL() << llvm::toString(crossed.takeError());
  payload = findPayload(*crossed);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->costModelKind, StageCostModelKind::IndirectGatherMemory);
  EXPECT_FALSE(payload->features.hasPartialContinuousMemory);
  EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadRows, 0.0);
  EXPECT_DOUBLE_EQ(payload->workload.indirectLoadBytes, 256.0);

  // With both prefix axes unstructured, both expand and each structured
  // slice contains the final 8 elements.
  replaceOnce("arith.muli %outer_broadcast, %row_stride",
              "arith.muli %index_broadcast, %row_stride");
  auto twoIndirectAxes =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  ASSERT_TRUE(twoIndirectAxes);
  auto twoAxisPartition = StagePartitioner().partition(
      *twoIndirectAxes, mlir::ascend::SimtAnchorPlan{},
      StagePartitionerOptions{});
  if (!twoAxisPartition)
    FAIL() << llvm::toString(twoAxisPartition.takeError());
  payload = findPayload(*twoAxisPartition);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->costModelKind,
            StageCostModelKind::PartialContinuousTileMemory);
  EXPECT_TRUE(payload->features.hasPartialContinuousMemory);
  EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadRows, 8.0);
  EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadBytes, 256.0);
  EXPECT_DOUBLE_EQ(payload->workload.partialContinuousLoadWarpInstructions,
                   8.0);
  EXPECT_DOUBLE_EQ(payload->workload.indirectLoadTransactions, 0.0);
}

TEST(StageAddressPatternTest, LoadedScalarIndexRemainsIndirect) {
  for (bool store : {false, true}) {
    std::string source = partialContinuousTileIR(8, 8, 1, store);
    size_t begin = source.find("        %index = tt.load");
    size_t end = source.find("        %index2d =", begin);
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    // A loaded scalar is uniform across both axes. Pointer analysis reports
    // no unstructured axes, but the original loaded-index rule still applies.
    source.replace(begin, end - begin,
                   "        %shift = tt.load %indices : !tt.ptr<i32>\n"
                   "        %index = tt.splat %shift : i32 -> tensor<8xi32>\n");
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::triton::TritonDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto partition = StagePartitioner().partition(
        *module, mlir::ascend::SimtAnchorPlan{}, StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    bool sawPayload = false;
    for (const LogicalStage &stage : partition->stages)
      for (mlir::Operation *operation : stage.operations) {
        if (operation->getName().getStringRef() !=
            (store ? "tt.store" : "tt.load"))
          continue;
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(
            operation->getOperand(0).getType());
        if (!type || type.getShape() != llvm::ArrayRef<int64_t>({8, 8}))
          continue;
        sawPayload = true;
        EXPECT_TRUE(mlir::ascend::isLoadedIndexDependentMemoryOp(operation));
        EXPECT_EQ(stage.costModelKind,
                  StageCostModelKind::IndirectGatherMemory);
        EXPECT_TRUE(stage.features.hasIndirectMemory);
        EXPECT_FALSE(stage.features.hasPartialContinuousMemory);
        EXPECT_DOUBLE_EQ(store ? stage.workload.indirectStoreBytes
                               : stage.workload.indirectLoadBytes,
                         256.0);
        EXPECT_DOUBLE_EQ(store ? stage.workload.partialContinuousStoreRows
                               : stage.workload.partialContinuousLoadRows,
                         0.0);
      }
    EXPECT_TRUE(sawPayload);
  }
}
