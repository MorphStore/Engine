/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file column_cache.h
 * @brief 
 * @todo Documentation.
 * @todo Do not use `std::unordered_map::at`, since it might throw exceptions
 * whose messages are hard to interpret. Throw understandable exceptions.
 */

#ifndef MORPHSTORE_CORE_UTILS_COLUMN_CACHE_H
#define MORPHSTORE_CORE_UTILS_COLUMN_CACHE_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/storage/column.h>
#include <core/utils/processing_style.h>
#include <core/utils/preprocessor.h>

#include <iostream>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>


namespace morphstore {

    class column_cache {
        
        class abstract_column_wrapper {
        protected:
            abstract_column_wrapper(const void * p_ColumnVoidPtr)
            : m_ColumnVoidPtr(p_ColumnVoidPtr) {
                //
            };
            
        public:
            const void * const m_ColumnVoidPtr;
            
            virtual ~abstract_column_wrapper() = default;
        };
        
        template<class t_format>
        class column_wrapper : public abstract_column_wrapper {
        public:
            column_wrapper(const column<t_format> * p_Col)
            : abstract_column_wrapper(reinterpret_cast<const void *>(p_Col)) {
                //
            }
            
            ~column_wrapper() {
                delete reinterpret_cast<const column<t_format> *>(
                        m_ColumnVoidPtr
                );
            }
        };
        
        std::unordered_map<
                const column<uncompr_f> *,
                std::unordered_map<
                        std::type_index,
                        abstract_column_wrapper *
                >
        > m_Cache;
        
    public:
        column_cache() = default;
        
        column_cache(const column_cache &) = delete;
        column_cache(column_cache &&) = delete;
        column_cache & operator=(const column_cache &) = delete;
        column_cache & operator=(column_cache &&) = delete;
        
        ~column_cache() {
            clear();
        }
        
        // We need some return value, so that we can use this function in
        // parameter pack expansions.
        template<class t_format>
        bool ensure_presence(const column<uncompr_f> * p_Col) {
            auto & formatMap = m_Cache[p_Col];
            std::type_index formatKey(typeid(t_format));
            if(!formatMap.count(formatKey)) {
                // @todo Do not hardcode the processing style here.
                formatMap.emplace(
                        formatKey,
                        new column_wrapper<t_format>(
                                morph<processing_style_t::scalar, t_format>(
                                        p_Col
                                )
                        )
                );
                return false;
            }
            else
                return true;
        }
        
        template<class t_format>
        const column<t_format> * get(const column<uncompr_f> * p_Col) const {
            return reinterpret_cast<const column<t_format> *>(
                    m_Cache.at(p_Col).at(
                            std::type_index(typeid(t_format))
                    )->m_ColumnVoidPtr
            );
        }
        
        void clear() {
            for(auto itCol : m_Cache)
                for(auto itFormat : itCol.second)
                    delete itFormat.second;
            m_Cache.clear();
        }
        
        void clear(const column<uncompr_f> * p_Col) {
            auto & formatMap = m_Cache.at(p_Col);
            for(auto itFormat : formatMap)
                delete itFormat.second;
            m_Cache.erase(p_Col);
        }
        
        void print(std::ostream & os) const {
            for(auto itCol : m_Cache) {
                os << reinterpret_cast<const void *>(itCol.first) << std::endl;
                for(auto itFormat : itCol.second)
                    os
                            << '\t' << itFormat.first.name() << ": "
                            << reinterpret_cast<const void *>(itFormat.second)
                            << std::endl;
            }
        }
    };
    
    template<>
    bool column_cache::ensure_presence<uncompr_f>(
            MSV_CXX_ATTRIBUTE_PPUNUSED const column<uncompr_f> * p_Col
    ) {
        return false;
    }
        
    template<>
    const column<uncompr_f> * column_cache::get(
            const column<uncompr_f> * p_Col
    ) const {
        return p_Col;
    }
    
    std::ostream & operator<<(std::ostream & os, const column_cache & cc) {
        cc.print(os);
        return os;
    }

}
#endif //MORPHSTORE_CORE_UTILS_COLUMN_CACHE_H
