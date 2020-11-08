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

#ifndef PIXIENN_SINGLETONREGISTRY_H
#define PIXIENN_SINGLETONREGISTRY_H

#include "Error.h"
#include "Singleton.h"

#include <shared_mutex>
#include <unordered_map>

namespace px {

template<typename Key, typename Value>
class SingletonRegistry : public Singleton<SingletonRegistry<Key, Value>>
{
public:
    const Value* add(const Key& key, std::unique_ptr<Value>&& value);
    const Value* find(const Key& key) const;
    std::vector<Key> keys() const;

    ~SingletonRegistry();

private:
    friend class Singleton<SingletonRegistry<Key, Value>>;

    std::unordered_map<Key, Value*> container_;
    mutable std::shared_mutex mutex_;
};

template<typename Key, typename Value>
const Value* SingletonRegistry<Key, Value>::add(const Key& key, std::unique_ptr<Value>&& value)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto v = std::move(value);
    DG_CHECK(v != nullptr, "Registry entries cannot be NULL");

    auto it = container_.find(key);
    if (it != container_.end()) {
        return it->second;
    }

    auto result = container_.emplace(key, v.release());
    return result.first->second;
}

template<typename Key, typename Value>
const Value* SingletonRegistry<Key, Value>::find(const Key& key) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = container_.find(key);
    if (it != container_.end()) {
        return it->second;
    }

    return nullptr;
}

template<typename Key, typename Value>
std::vector<Key> SingletonRegistry<Key, Value>::keys() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<Key> k;

    for (const auto& item : container_) {
        k.push_back(item.first);
    }

    return k;
}

template<typename Key, typename Value>
SingletonRegistry<Key, Value>::~SingletonRegistry()
{
    for (auto& item : container_) {
        delete item.second;
    }
}

} // px

#endif // PIXIENN_SINGLETONREGISTRY_H
