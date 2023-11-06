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

#ifndef PIXIENN_TIMER_H
#define PIXIENN_TIMER_H

#include <chrono>
#include "Common.h"

namespace px {

class Timer
{
public:
    Timer();
    ~Timer();

    [[nodiscard]] std::string str() const;
    void restart();
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TimePoint start_;
};

inline std::ostream& operator<<(std::ostream& s, const Timer& timer)
{
    return s << timer.str();
}

}   // px

#endif // PIXIENN_TIMER_H
