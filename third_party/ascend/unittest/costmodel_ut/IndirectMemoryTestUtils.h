#ifndef ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H
#define ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H

#include "CostModelTestUtils.h"

namespace mlir::ascend::test {

/// Construct the same per-axis evidence for load/store domain checks.
/// Keep special axes, masks, reuse proofs and expected prices in each test.
inline AddressPatternSummary
loadedAddressPattern(const LogicalStage &stage, llvm::StringRef memoryOp,
                     llvm::ArrayRef<int64_t> shape,
                     llvm::StringRef regularity = "opaque_loaded",
                     bool dependsOnLoadedValue = true) {
  AddressPatternSummary pattern;
  pattern.stageId = stage.id;
  pattern.memoryOp = memoryOp.str();
  pattern.dependsOnLoadedValue = dependsOnLoadedValue;
  for (int64_t extent : shape) {
    AddressAxisSummary axis;
    axis.extent = extent;
    axis.regularity = regularity.str();
    pattern.axes.push_back(axis);
  }
  return pattern;
}

} // namespace mlir::ascend::test

#endif // ASCEND_COSTMODEL_UT_INDIRECTMEMORYTESTUTILS_H
