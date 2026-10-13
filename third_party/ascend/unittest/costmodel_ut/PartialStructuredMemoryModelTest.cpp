#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"
#include <gtest/gtest.h>

using namespace mlir::ascend::partial_memory;

TEST(PartialStructuredMemoryModelTest, FrozenReferencePredictions) {
  // References from frozen Python inference; asymmetric shape and large odd
  // stride exercise byte geometry and histogram threshold equality.
  Geometry ds{{{2, 15346, true}, {16, 1023, false}}, 32, 1, true};
  for (auto item : {std::pair{Model::SimtLoad, 184.28974087203204},
                    std::pair{Model::SimtStore, 206.82532932691726},
                    std::pair{Model::SimdLoad, 157.83632226295097}}) {
    auto result = predict(ds, item.first);
    ASSERT_TRUE(result);
    EXPECT_NEAR(*result, item.second, 1e-8);
  }
  Geometry sd{{{2, 49088, false}, {16, 1, true}}, 32, 1, true};
  auto result = predict(sd, Model::SimdStore);
  ASSERT_TRUE(result);
  EXPECT_NEAR(*result, 2976.733291731354, 1e-8);
}

TEST(PartialStructuredMemoryModelTest, UnsupportedDomainFallsBack) {
  Geometry valid{{{4, 8, true}, {8, 1, false}}, 32, 1, true};
  ASSERT_TRUE(predict(valid, Model::SimtLoad));
  auto q2 = valid;
  q2.warps = 2;
  EXPECT_FALSE(predict(q2, Model::SimtLoad));
  auto zero = valid;
  zero.axes.back().pitch = 0;
  EXPECT_FALSE(predict(zero, Model::SimtLoad));
  auto negative = valid;
  negative.axes.back().pitch = -1;
  EXPECT_FALSE(predict(negative, Model::SimdStore));
  auto allD = valid;
  allD.axes.back().discrete = true;
  EXPECT_FALSE(predict(allD, Model::SimtLoad));
  auto allS = valid;
  allS.axes.front().discrete = false;
  EXPECT_FALSE(predict(allS, Model::SimdLoad));
  auto boolean = valid;
  boolean.bits = 1;
  EXPECT_FALSE(predict(boolean, Model::SimdStore));
  auto nonPower = valid;
  nonPower.warps = 3;
  EXPECT_FALSE(predict(nonPower, Model::SimtStore));
}

TEST(PartialStructuredMemoryModelTest,
     DiscretePitchIsNotRuntimeAddressCoverage) {
  Geometry g{{{4, 32, true}, {8, 4, false}}, 16, 1, true};
  auto f = extractFeatures(g);
  EXPECT_DOUBLE_EQ(f.at("log_max_D_pitch"), 5);
  EXPECT_DOUBLE_EQ(f.at("log_tail_step_bytes"), 3);
  EXPECT_DOUBLE_EQ(f.at("static_work"), 32);
  EXPECT_EQ(f.count("U"), 0u);
  EXPECT_EQ(f.count("V"), 0u);
}
