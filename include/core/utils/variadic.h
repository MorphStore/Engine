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
 * @file variadic.h
 * @brief Some utilities for working with variadic templates.
 */

#ifndef MORPHSTORE_CORE_UTILS_VARIADIC_H
#define MORPHSTORE_CORE_UTILS_VARIADIC_H

#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <functional>
#include <core/utils/preprocessor.h>
#include <core/utils/helper_types.h>

    
#define STATIC_ASSERT_PARAMPACK_SAMESIZE(pack1, pack2) \
static_assert( \
        sizeof...(pack1) == sizeof...(pack2), \
        "template parameter pack " #pack1 " should have the same size as " \
        "template parameter pack " #pack2 \
);

namespace morphstore {

    template<
            unsigned t_Count,
            typename t_type,
            t_type t_Value,
            t_type ... t_MoreValues
    >
    struct repeat_as_tuple : public repeat_as_tuple<
            t_Count - 1, t_type, t_Value, t_Value, t_MoreValues ...
    > {
        //
    };

    template<typename t_type, t_type t_Value, t_type ... t_MoreValues>
    struct repeat_as_tuple<1, t_type, t_Value, t_MoreValues ...> {
        static constexpr auto value = std::make_tuple(
                t_Value, t_MoreValues ...
        );
    };
    
    // fold-expressions only in c++1z / c++17
    template<typename ... T>
    std::string doPrint(char delim, T ... values) {
        std::stringstream ss;
        using isoHelper = int[];
        (void)isoHelper {
                0, (void(ss << std::forward< T >(values) << delim), 0) ...
        };
        return ss.str();
    }
    
    // fold-expressions only in c++1z / c++17
    template<typename ... T>
    std::string doPrintWithWidths(const std::string & delim, const size_t * ws, T ... values) {
        std::stringstream ss;
        using isoHelper = int[];
        unsigned i = 0;
        (void)isoHelper {
                0, (void(ss << std::left << std::setw(ws[i++]) << std::forward< T >(values) << delim), 0) ...
        };
        return ss.str();
    }


   template<std::size_t I = 0, typename Tuple, typename Func>
   constexpr
   typename std::enable_if< I != std::tuple_size<Tuple>::value, void >::type
   static_for_each(const Tuple& p_Tuple, Func&& p_Func) {
      p_Func(std::get<I>(p_Tuple));
      static_for_each<I + 1>(p_Tuple, p_Func);
   }

   template<std::size_t I = 0, typename Tuple, typename Func>
   constexpr
   typename std::enable_if< I == std::tuple_size<Tuple>::value, void >::type
   static_for_each(const Tuple& MSV_CXX_ATTRIBUTE_PPUNUSED p_Tuple, Func&& MSV_CXX_ATTRIBUTE_PPUNUSED p_Func){}


   template< int Alignment, typename T, T... Ns >
   constexpr auto create_sequence_array( T const start, T const stepwidth, std::integer_sequence< T, Ns... > ) -> aligned_array< Alignment, T, sizeof...(Ns) > {
      return {( Ns * stepwidth + start )...};
   }

   template< int Alignment, typename T, T N >
   constexpr auto static_sequence_array( T const start = 0, T const stepwidth = 1 ) noexcept {
      return create_sequence_array<Alignment, T >( start, stepwidth, std::make_integer_sequence< T, N >{} );
   }

   template< typename... Ts >
   struct static_and : std::true_type { };
   template< typename T, typename... Ts >
   struct static_and< T, Ts... > :
      std::conditional< T::value, static_and< Ts... >, std::false_type >::type{ };
   template< typename T, typename... Ts >
   using static_all_T = static_and< std::is_same< Ts, T >... >;


}
#endif //MORPHSTORE_CORE_UTILS_VARIADIC_H
