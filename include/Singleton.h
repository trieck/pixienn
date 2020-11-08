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

#ifndef PIXIENN_SINGLETON_H
#define PIXIENN_SINGLETON_H

#include <atomic>
#include <mutex>
#include <utility>

namespace px {
template<class T>
class Singleton
{
public:
    static T& instance()
    {
        auto instance = instance_.load(std::memory_order_acquire);
        if (!instance) {
            std::lock_guard<std::mutex> lock(mutex_);
            instance = instance_.load(std::memory_order_relaxed);
            if (!instance) {
                instance = new T();
                instance_.store(instance, std::memory_order_release);
            }
        }

        return *instance;
    }

private:
    static std::mutex mutex_;
    static std::atomic<T*> instance_;
};

template<class T>
std::mutex Singleton<T>::mutex_;

template<class T>
std::atomic<T*> Singleton<T>::instance_ = ATOMIC_VAR_INIT(nullptr);

} // px

#endif // PIXIENN_SINGLETON_H
