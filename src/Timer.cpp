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

#include "Timer.h"
#include <boost/format.hpp>

namespace px {

Timer::Timer()
{
    start_ = Clock::now();
}

Timer::~Timer() = default;

std::string Timer::str() const
{
    using std::chrono::duration_cast;

    auto now = Clock::now();

    auto elapsed = duration_cast<std::chrono::nanoseconds>(now - start_);
    auto hours = duration_cast<std::chrono::hours>(elapsed);
    auto minutes = duration_cast<std::chrono::minutes>(elapsed) % 3600 / 60;
    auto seconds = duration_cast<std::chrono::seconds>(elapsed) % 60;
    auto millis = duration_cast<std::chrono::milliseconds>(elapsed) % 1000;
    auto micros = duration_cast<std::chrono::microseconds>(elapsed) % 1000;
    auto nanos = elapsed % 1000;

    boost::format fmt;

    if (hours.count())
        fmt = boost::format("%d:%02d:%02d hours") % hours.count() % minutes.count() % seconds.count();
    else if (minutes.count())
        fmt = boost::format("%d:%02d minutes") % minutes.count() % seconds.count();
    else if (millis.count())
        fmt = boost::format("%d.%03d seconds") % seconds.count() % millis.count();
    else
        fmt = boost::format("%d.%03d microseconds") % micros.count() % nanos.count();

    return fmt.str();
}

void Timer::restart()
{
    start_ = Clock::now();
}

}   // px
