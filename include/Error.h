/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#ifndef PIXIENN_ERROR_H
#define PIXIENN_ERROR_H

#include "Common.h"

namespace px {

class Error : public std::exception
{
public:
    Error() noexcept;
    Error(const Error& error, const std::string& message) noexcept;
    Error(const char* file, unsigned int line, const char* function, const char* message) noexcept;
    Error(const char* file, unsigned int line, const char* function, const std::string& message) noexcept;
    Error(const std::string& file, unsigned int line, const std::string& function, const std::string& message) noexcept;
    Error(const char* file, unsigned int line, const char* function, const std::exception_ptr& ptr, const char* format,
          ...) noexcept;

    Error(const Error& rhs) noexcept;
    Error& operator=(const Error& rhs) noexcept;

    static Error
    fromFormat(const char* file, unsigned int line, const char* function, const char* format, ...) noexcept;

    // std::exception
    virtual const char* what() const noexcept override;

    const std::string& file() const noexcept;
    unsigned int line() const noexcept;
    const std::string& function() const noexcept;
    const std::string& message() const noexcept;

protected:
    void init() noexcept;

    std::string file_;
    uint32_t line_ = 0;
    std::string function_;
    std::string message_;
    std::string what_;
};

#ifndef __FILENAME__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#endif // __FILENAME__

#define PX_ERROR_THROW(format, ...) throw px::Error::fromFormat(__FILENAME__, __LINE__, __FUNCTION__, format, ##__VA_ARGS__)
#define PX_CHECK(condition, message, ...) if(!(condition)) PX_ERROR_THROW((message), ##__VA_ARGS__)

} // px

#endif // PIXIENN_ERROR_H
