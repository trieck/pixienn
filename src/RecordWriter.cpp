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

#include "Common.h"
#include "CRC32.h"
#include "Error.h"
#include "RecordWriter.h"

namespace px {

static constexpr size_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
static constexpr size_t kFooterSize = sizeof(uint32_t);

static void encodeFixed64(char* buf, uint64_t value)
{
    value = boost::endian::native_to_little(value);
    std::memcpy(buf, &value, sizeof(value));
}

static void encodeFixed32(char* buf, uint32_t value)
{
    value = boost::endian::native_to_little(value);
    std::memcpy(buf, &value, sizeof(value));
}

RecordWriter::RecordWriter(const std::string& filename, bool append)
{
    open(filename, append);
    PX_CHECK(file_.is_open(), "Could not open file");
}

RecordWriter::~RecordWriter()
{
    close();
}

RecordWriter::Ptr RecordWriter::create(const std::string& filename, bool append)
{
    return std::make_unique<RecordWriter>(filename, append);
}

void RecordWriter::write(const tensorflow::Event& event)
{
    std::string record;
    event.SerializeToString(&record);

    write(record);
}

void RecordWriter::write(const std::string& record)
{
    // Format of a single record:
    //  uint64    length
    //  uint32    masked crc of length
    //  byte      data[length]
    //  uint32    masked crc of data

    char header[kHeaderSize];
    char footer[kFooterSize];

    populateHeader(header, record.data(), record.size());
    populateFooter(footer, record.data(), record.size());

    PX_CHECK(file_.good(), "Could not write record");

    file_.write(header, sizeof(header));
    file_.write(record.data(), record.size());
    file_.write(footer, sizeof(footer));
}

void RecordWriter::populateHeader(char* header, const char* data, size_t size)
{
    encodeFixed64(header + 0, size);
    encodeFixed32(header + sizeof(uint64_t), MaskedCrc(header, sizeof(uint64_t)));
}

void RecordWriter::populateFooter(char* footer, const char* data, size_t size)
{
    encodeFixed32(footer, MaskedCrc(data, size));
}

void RecordWriter::open(const std::string& filename, bool append)
{
    close();

    auto flags = std::ios::out | std::ios::binary;
    if (append) {
        flags |= std::ios::app;
    }

    file_.open(filename, flags);
    PX_CHECK(file_.is_open(), "Could not open file \"%s\"", filename.c_str());
}

void RecordWriter::close()
{
    if (file_.is_open()) {
        file_.close();
    }
}

}   // px

