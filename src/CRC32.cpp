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

#include <boost/endian/conversion.hpp>

#include "CRC32.h"
#include "Error.h"
#include "Singleton.h"

static constexpr int ZEROES_BASE_LG = 4;
static constexpr int ZEROES_BASE = 1 << ZEROES_BASE_LG;

namespace px {

static void PolyMultiply(uint32_t* val, uint32_t m, uint32_t poly)
{
    uint32_t l = *val;
    uint32_t result = 0;
    auto onebit = uint32_t{ 0x80000000u };
    for (uint32_t one = onebit; one != 0; one >>= 1) {
        if ((l & one) != 0) {
            result ^= m;
        }
        if (m & 1) {
            m = (m >> 1) ^ poly;
        } else {
            m >>= 1;
        }
    }
    *val = result;
}

static uint32_t ReverseBits(uint32_t bits)
{
    bits = (bits & 0xaaaaaaaau) >> 1 | (bits & 0x55555555u) << 1;
    bits = (bits & 0xccccccccu) >> 2 | (bits & 0x33333333u) << 2;
    bits = (bits & 0xf0f0f0f0u) >> 4 | (bits & 0x0f0f0f0fu) << 4;
    return boost::endian::endian_reverse(bits);
}

///////////////////////////////////////////////////////////////////////////////
class CRC32Engine : public Singleton<CRC32Engine>
{
public:
    CRC32Engine();
    ~CRC32Engine();

    void extend(uint32_t* crc, const char* data, size_t n) const;

private:
    using Uint32By256 = uint32_t[256];

    void initTables();

    void fillWordTable(uint32_t poly, uint32_t last, int wordSize, Uint32By256* t);
    int fillZeroesTable(uint32_t poly, Uint32By256* t);

    static constexpr uint32_t kCrc32cPoly = 0x82f63b78;

    uint32_t table0_[256];          // table of byte extensions
    uint32_t zeroes_[256];          // table of zero extensions

    uint32_t table_[4][256];        // table of 4-byte extensions shifted by 12 bytes of zeroes

    uint32_t reverse_table0_[256];  // table of reverse byte extensions
    uint32_t reverse_zeroes_[256];  // table of reverse zero extensions
};

CRC32Engine::CRC32Engine()
{
    initTables();
}

CRC32Engine::~CRC32Engine()
{
}

void CRC32Engine::initTables()
{
    auto t = std::make_unique<Uint32By256[]>(4);

    fillWordTable(kCrc32cPoly, kCrc32cPoly, 1, t.get());

    for (auto i = 0; i < 256; ++i) {
        this->table0_[i] = t[0][i];
    }

    // Construct a table for updating the CRC by 4 bytes data followed by
    // 12 bytes of zeroes.
    //
    // Note: the data word size could be larger than the CRC size; it might
    // be slightly faster to use a 64-bit data word, but doing so doubles the
    // table size.
    auto last = kCrc32cPoly;
    const size_t size = 12;
    for (size_t i = 0; i < size; ++i) {
        last = (last >> 8) ^ this->table0_[last & 0xff];
    }

    fillWordTable(kCrc32cPoly, last, 4, t.get());

    for (size_t b = 0; b < 4; ++b) {
        for (size_t i = 0; i < 256; ++i) {
            this->table_[b][i] = t[b][i];
        }
    }

    auto j = fillZeroesTable(kCrc32cPoly, t.get());
    PX_CHECK(j <= 256, "Overflowed zeroes table");

    for (auto i = 0; i < j; ++i) {
        this->zeroes_[i] = t[0][i];
    }

    t.reset();

    const auto kCrc32cUnextendPoly = ReverseBits(static_cast<uint32_t>((kCrc32cPoly << 1) ^ 1));
    fillWordTable(kCrc32cUnextendPoly, kCrc32cUnextendPoly, 1, &reverse_table0_);

    j = fillZeroesTable(kCrc32cUnextendPoly, &reverse_zeroes_);
    PX_CHECK(j <= 256, "Overflowed reverse zeroes table");
}

void CRC32Engine::fillWordTable(uint32_t poly, uint32_t last, int wordSize, CRC32Engine::Uint32By256* t)
{
    for (auto j = 0; j != wordSize; j++) {   // for each byte of extension....
        t[j][0] = 0;                        // a zero has no effect
        for (auto i = 128; i != 0; i >>= 1) {  // fill in entries for powers of 2
            if (j == 0 && i == 128) {
                t[j][i] = last;  // top bit in last byte is given
            } else {
                // each successive power of two is derived from the previous
                // one, either in this table, or the last table
                uint32_t pred;
                if (i == 128) {
                    pred = t[j - 1][1];
                } else {
                    pred = t[j][i << 1];
                }
                // Advance the CRC by one bit (multiply by X, and take remainder
                // through one step of polynomial long division)
                if (pred & 1) {
                    t[j][i] = (pred >> 1) ^ poly;
                } else {
                    t[j][i] = pred >> 1;
                }
            }
        }
        // CRCs have the property that CRC(a xor b) == CRC(a) xor CRC(b)
        // so we can make all the tables for non-powers of two by
        // xoring previously created entries.
        for (auto i = 2; i != 256; i <<= 1) {
            for (auto k = i + 1; k != (i << 1); k++) {
                t[j][k] = t[j][i] ^ t[j][k - i];
            }
        }
    }
}

int CRC32Engine::fillZeroesTable(uint32_t poly, CRC32Engine::Uint32By256* t)
{
    uint32_t inc = 1 << 31;

    // Extend by one zero bit. We know degree > 1 so (inc & 1) == 0.
    inc >>= 1;

    // Now extend by 2, 4, and 8 bits, so now `inc` is extended by one zero byte.
    for (auto i = 0; i < 3; ++i) {
        PolyMultiply(&inc, inc, poly);
    }

    auto j = 0;
    for (uint64_t incLen = 1; incLen != 0; incLen <<= ZEROES_BASE_LG) {
        // Every entry in the table adds an additional incLen zeroes.
        uint32_t v = inc;
        for (int a = 1; a != ZEROES_BASE; a++) {
            t[0][j] = v;
            PolyMultiply(&v, inc, poly);
            j++;
        }
        inc = v;
    }

    PX_CHECK(j <= 256, "Overflowed zeroes table");

    return j;
}

void CRC32Engine::extend(uint32_t* crc, const char* data, size_t n) const
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    const auto* e = p + n;
    uint32_t l = *crc;

    const size_t kSwathSize = 16;
    if (static_cast<size_t>(e - p) >= kSwathSize) {
        // Load one swath of data into the operating buffers.
        auto buf0 = boost::endian::load_little_u32(p) ^ l;
        auto buf1 = boost::endian::load_little_u32(p + 4);
        auto buf2 = boost::endian::load_little_u32(p + 8);
        auto buf3 = boost::endian::load_little_u32(p + 12);

        p += kSwathSize;

        // Increment a CRC value by a "swath"; this combines the four bytes
        // starting at `ptr` and twelve zero bytes, so that four CRCs can be
        // built incrementally and combined at the end.
        const auto stepSwath = [this](uint32_t crc_in, const std::uint8_t* ptr) {
            return boost::endian::load_little_u32(ptr) ^
                   this->table_[3][crc_in & 0xff] ^
                   this->table_[2][(crc_in >> 8) & 0xff] ^
                   this->table_[1][(crc_in >> 16) & 0xff] ^
                   this->table_[0][crc_in >> 24];
        };

        // Run one CRC calculation step over all swaths in one 16-byte stride
        const auto stepStride = [&]() {
            buf0 = stepSwath(buf0, p);
            buf1 = stepSwath(buf1, p + 4);
            buf2 = stepSwath(buf2, p + 8);
            buf3 = stepSwath(buf3, p + 12);
            p += 16;
        };

        static constexpr int kPrefetchHorizon = 64 * 4;

        // Process kStride interleaved swaths through the data in parallel.
        while ((e - p) > kPrefetchHorizon) {
            // Process 64 bytes at a time
            stepStride();
            stepStride();
            stepStride();
            stepStride();
        }

        while (static_cast<size_t>(e - p) >= kSwathSize) {
            stepStride();
        }

        // Now advance one word at a time as far as possible. This isn't worth
        // doing if we have word-advance tables.
        while (static_cast<size_t>(e - p) >= 4) {
            buf0 = stepSwath(buf0, p);
            uint32_t tmp = buf0;
            buf0 = buf1;
            buf1 = buf2;
            buf2 = buf3;
            buf3 = tmp;
            p += 4;
        }

        // Combine the results from the different swaths. This is just a CRC
        // on the data values in the bufX words.
        auto combineOneWord = [this](uint32_t crc_in, uint32_t w) {
            w ^= crc_in;
            for (size_t i = 0; i < 4; ++i) {
                w = (w >> 8) ^ this->table0_[w & 0xff];
            }
            return w;
        };

        l = combineOneWord(0, buf0);
        l = combineOneWord(l, buf1);
        l = combineOneWord(l, buf2);
        l = combineOneWord(l, buf3);
    }

    auto stepOneByte = [this, &p, &l]() {
        int c = (l & 0xff) ^ *p++;
        l = this->table0_[c] ^ (l >> 8);
    };

    // Process the last few bytes
    while (p != e) {
        stepOneByte();
    }

    *crc = l;

}

///////////////////////////////////////////////////////////////////////////////

CRC32::CRC32()
{
    engine_ = &CRC32Engine::instance();
}

CRC32::~CRC32()
{
}

uint32_t CRC32::operator()(const char* data, size_t n) const
{
    return extend(0, data, n);
}

uint32_t CRC32::extend(uint32_t initialCrc, const char* data, size_t n) const
{
    static constexpr uint32_t kCRC32Xor = 0xffffffffU;

    auto crc = initialCrc ^ kCRC32Xor;

    engine_->extend(&crc, data, n);

    crc = crc ^ kCRC32Xor;

    return crc;
}

static uint32_t Mask(uint32_t crc)
{
    static const uint32_t kMaskDelta = 0xa282ead8ul;

    // Rotate right by 15 bits and add a constant.
    return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

uint32_t MaskedCrc(const char* data, size_t n)
{
    auto value = CRC32()(data, n);

    auto mask = Mask(value);

    return mask;
}

}   // px
