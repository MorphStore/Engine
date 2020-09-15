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
 * @file equality_check.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_EQUALITY_CHECK_H
#define MORPHSTORE_CORE_UTILS_EQUALITY_CHECK_H

#include <core/morphing/format.h>
#include <core/storage/column.h>

#include <algorithm>
#include <cstring>
#include <iostream>

#include <assert.h> 

namespace morphstore {

    struct equality_check {
        const size_t m_CountValuesExp;
        const size_t m_CountValuesFnd;
        const bool m_CountValuesEqual;
        const size_t m_SizeUsedByteExp;
        const size_t m_SizeUsedByteFnd;
        const bool m_SizeUsedByteEqual;
        const bool m_DataEqual;
        
        template< class F >
        equality_check(
                const column< F > * colExp,
                const column< F > * colFnd
        ) :
            m_CountValuesExp( colExp->get_count_values( ) ),
            m_CountValuesFnd( colFnd->get_count_values( ) ),
            m_CountValuesEqual( m_CountValuesExp == m_CountValuesFnd ),
            m_SizeUsedByteExp( colExp->get_size_used_byte( ) ),
            m_SizeUsedByteFnd( colFnd->get_size_used_byte( ) ),
            m_SizeUsedByteEqual( m_SizeUsedByteExp == m_SizeUsedByteFnd ),
            m_DataEqual( !memcmp(
                    colExp->get_data( ),
                    colFnd->get_data( ),
                    std::min( m_SizeUsedByteExp, m_SizeUsedByteFnd )
            ) )
        {
            //
        }
        
        bool good( ) const {
            return m_CountValuesEqual && m_SizeUsedByteEqual && m_DataEqual;
        }
        
        equality_check( equality_check const & ) = delete;
        equality_check( equality_check && ) = delete;
        equality_check & operator=( equality_check const & ) = delete;
        equality_check & operator=( equality_check && ) = delete;
        
        static const char * ok_str( bool ok ) {
            return ok ? "ok" : "not ok";
        }
    };
    
    std::ostream & operator<<( std::ostream & os, const equality_check & ec ) {
        const char *data_ok_str =
            (ec.m_CountValuesEqual && ec.m_SizeUsedByteEqual) ? equality_check::ok_str(ec.m_DataEqual) : "undefined";

        os
                << "countValues: " << equality_check::ok_str( ec.m_CountValuesEqual )
                << " (expected " << ec.m_CountValuesExp
                << ", found " << ec.m_CountValuesFnd << ')'
                << std::endl
                << "sizeUsedByte: " << equality_check::ok_str( ec.m_SizeUsedByteEqual )
                << " (expected " << ec.m_SizeUsedByteExp
                << ", found " << ec.m_SizeUsedByteFnd << ')'
                << std::endl
                << "data: " << data_ok_str
                << std::endl;
        return os;
    }

    template <class F> void assert_columns_equal(const column<F> *expected_col, const column<F> *actual_col) {
        equality_check ec(expected_col, actual_col);
        std::cout << ec;
        if (!ec.good()) {
            uint64_t *expected = expected_col->get_data();
            uint64_t *actual = actual_col->get_data();

            assert(ec.m_CountValuesEqual);
            assert(ec.m_SizeUsedByteEqual);

            // printing only different entries
            for (uint64_t i = 0; i < expected_col->get_count_values(); i++) {
                if (!(expected[i] == actual[i])) {
                    std::cout << "pos: " << i << " expected: " << expected[i] << " actual: " << actual[i] << std::endl;
                }
            }
            // print_columns(print_buffer_base::decimal, actual_col, expected_col, "actual", "expected");
            assert(false);
        }
    }
}
#endif //MORPHSTORE_CORE_UTILS_EQUALITY_CHECK_H
