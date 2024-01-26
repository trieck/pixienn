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

#pragma once

#include "event.pb.h"

namespace px {

class RecordWriter
{
public:
    using Ptr = std::unique_ptr<RecordWriter>;

    RecordWriter() = default;
    RecordWriter(const std::string& filename, bool append = false);
    RecordWriter(const RecordWriter&) = delete;
    ~RecordWriter();

    static Ptr create(const std::string& filename, bool append = false);

    RecordWriter& operator=(const RecordWriter&) = delete;

    void write(const std::string& record);
    void write(const tensorflow::Event& event);

    void open(const std::string& filename, bool append = false);
    void close();

private:
    void populateHeader(char* header, const char* data, size_t size);
    void populateFooter(char* footer, const char* data, size_t size);

    std::ofstream file_;
};

}   // px
