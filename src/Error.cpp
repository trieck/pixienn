/********************************************************************************
* Copyright 2020 Thomas A. Rieck
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

#include "Error.h"
#include <cstdarg>

#define PX_ERROR_MAX_LEN (2048)

namespace px {

Error::Error() noexcept: line_(0)
{
    init();
}

Error::Error(const Error& error, const std::string& message) noexcept:
        file_(error.file_),
        line_(error.line_),
        function_(error.function_),
        message_(message)
{
    init();
}

Error::Error(const char* file, unsigned int line, const char* function, const char* message) noexcept:
        file_(file ? file : ""),
        line_(line),
        function_(function ? function : ""),
        message_(message ? message : "")
{
    init();
}

Error::Error(const std::string& file, unsigned int line, const std::string& function,
             const std::string& message) noexcept:
        file_(file),
        line_(line),
        function_(function),
        message_(message)
{
    init();
}

Error::Error(const char* file, unsigned int line, const char* function, const std::string& message) noexcept:
        file_(file ? file : ""),
        line_(line),
        function_(function ? function : ""),
        message_(message)
{
    init();
}

Error::Error(const char* file, unsigned int line, const char* function, const std::exception_ptr& ptr,
             const char* format, ...) noexcept
{
    std::string message;

    if (format != nullptr) {
        va_list args;
        va_start(args, format);

        char buf[PX_ERROR_MAX_LEN] = { 0 };
        vsnprintf(buf, PX_ERROR_MAX_LEN, format, args);

        va_end(args);

        message = buf;
    }

    try {
        std::rethrow_exception(ptr);
    } catch (const Error& e) {
        *this = Error(e, message + e.message());
    } catch (const std::exception& e) {
        *this = Error(file, line, function, message + e.what());
    } catch (...) {
        *this = Error(file, line, function, message + "Unknown exception");
    }
}

Error::Error(const Error& rhs) noexcept
{
    (void) operator=(rhs);
}

Error& Error::operator=(const Error& rhs) noexcept
{
    file_ = rhs.file_;
    line_ = rhs.line_;
    function_ = rhs.function_;
    message_ = rhs.message_;
    what_ = rhs.what_;

    return *this;
}

Error Error::fromFormat(const char* file, unsigned int line, const char* function, const char* format, ...) noexcept
{
    char buf[PX_ERROR_MAX_LEN] = { 0 };

    va_list args;
    va_start(args, format);
    vsnprintf(buf, PX_ERROR_MAX_LEN, format, args);
    va_end(args);

    return Error(file, line, function, buf);
}

const char* Error::what() const noexcept
{
    return what_.c_str();
}

const std::string& Error::file() const noexcept
{
    return file_;
}

unsigned int Error::line() const noexcept
{
    return line_;
}

const std::string& Error::function() const noexcept
{
    return function_;
}

const std::string& Error::message() const noexcept
{
    return message_;
}

void Error::init() noexcept
{
    std::string strLine = std::to_string(line_);
    what_ = message_ + " in " + function_ + ", file: " + file_ + ", line: " + strLine + ".";
}

}   // px
