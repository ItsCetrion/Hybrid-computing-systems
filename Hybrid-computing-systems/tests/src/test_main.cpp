#include <gtest/gtest.h>
#include "Core.h"

TEST(CoreTest, SumArray) {
    int data[] = { 1, 2, 3, 4, 5 };
    EXPECT_EQ(sumArray(data, 5), 16);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}