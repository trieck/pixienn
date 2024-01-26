/********************************************************************************
* Copyright 2020-2023 Thomas A. Rieck, All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "common.h"
#include "CRC32.h"

using namespace px;
using namespace testing;

TEST(CRC32, RFC3720)
{
    // Test the results of the vectors from
    // https://www.rfc-editor.org/rfc/rfc3720#appendix-B.4
    char data[32];

    // 32 bytes of zeros.
    memset(data, 0, sizeof(data));
    EXPECT_EQ(CRC32()(data, sizeof(data)), 0x8a9136aa);

    // 32 bytes of ones.
    memset(data, 0xff, sizeof(data));
    EXPECT_EQ(CRC32()(data, sizeof(data)), 0x62a8ab43);

    // 32 incrementing bytes.
    for (int i = 0; i < 32; ++i) data[i] = static_cast<char>(i);
    EXPECT_EQ(CRC32()(data, sizeof(data)), 0x46dd794e);

    // 32 decrementing bytes.
    for (int i = 0; i < 32; ++i) data[i] = static_cast<char>(31 - i);
    EXPECT_EQ(CRC32()(data, sizeof(data)), 0x113fdb5c);
}

TEST(CRC32, Compute)
{
    std::string data = "hello world";
    EXPECT_EQ(CRC32()(data.data(), data.size()), 0xc99465aa);
}