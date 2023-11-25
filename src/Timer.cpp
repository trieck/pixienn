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

    auto elapsed = duration_cast<std::chrono::milliseconds>(now - start_).count();

    auto hours = (elapsed / 1000) / 3600;
    auto minutes = ((elapsed / 1000) % 3600) / 60;
    auto seconds = (elapsed / 1000) % 60;
    auto millis = elapsed % 1000;

    boost::format fmt;

    if (hours)
        fmt = boost::format("%d:%02d:%02d hours") % hours % minutes % seconds;
    else if (minutes)
        fmt = boost::format("%d:%02d minutes") % minutes % seconds;
    else
        fmt = boost::format("%d.%03d seconds") % seconds % millis;

    return fmt.str();
}

void Timer::restart()
{
    start_ = Clock::now();
}

}   // px
