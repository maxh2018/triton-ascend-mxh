// Shared inputs reused from the existing SimdSimtCostModel test fixture.
#ifndef ASCEND_COSTMODEL_UT_COSTMODELTESTUTILS_H
#define ASCEND_COSTMODEL_UT_COSTMODELTESTUTILS_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include <gtest/gtest.h>

namespace mlir::ascend::test {

HardwareProfile hardwareProfile(StageTransitionCost transition = {});
LogicalStage
logicalStage(llvm::StringRef id, StageCostModelKind kind,
             StageScheduleKind schedule = StageScheduleKind::StraightLine,
             int64_t iterations = 1);
llvm::Expected<StageCostTable>
evaluateOneStage(LogicalStage stage,
                 HardwareProfile profile = hardwareProfile());

} // namespace mlir::ascend::test
#endif // ASCEND_COSTMODEL_UT_COSTMODELTESTUTILS_H
