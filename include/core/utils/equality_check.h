/**********************************************************************************************
 * Copyright (C) 2019 by Patrick Damme                                                        *
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
 * @author Patrick Damme
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_EQUALITY_CHECK_H
#define MORPHSTORE_CORE_UTILS_EQUALITY_CHECK_H

#include "../morphing/format.h"
#include "../storage/column.h"

#include <algorithm>
#include <cstring>
#include <iostream>

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
            m_CountValuesExp( colExp->count_values( ) ),
            m_CountValuesFnd( colFnd->count_values( ) ),
            m_CountValuesEqual( m_CountValuesExp == m_CountValuesFnd ),
            m_SizeUsedByteExp( colExp->size_used_byte( ) ),
            m_SizeUsedByteFnd( colFnd->size_used_byte( ) ),
            m_SizeUsedByteEqual( m_SizeUsedByteExp == m_SizeUsedByteFnd ),
            m_DataEqual( !memcmp(
                    colExp->data( ),
                    colFnd->data( ),
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
        
        static const char * okStr( bool ok ) {
            return ok ? "ok" : "not ok";
        }
    };
    
    std::ostream & operator<<( std::ostream & os, const equality_check & ec ) {
        os
                << "countValues: " << equality_check::okStr( ec.m_CountValuesEqual )
                << " (expected " << ec.m_CountValuesExp
                << ", found " << ec.m_CountValuesFnd << ')'
                << std::endl
                << "sizeUsedByte: " << equality_check::okStr( ec.m_SizeUsedByteEqual )
                << " (expected " << ec.m_SizeUsedByteExp
                << ", found " << ec.m_SizeUsedByteFnd << ')'
                << std::endl
                << "data: " << equality_check::okStr( ec.m_DataEqual )
                << " (this check is only valid, if countValues and sizeUsedByte are ok)"
                << std::endl;
        return os;
    }
}
#endif //MORPHSTORE_CORE_UTILS_EQUALITY_CHECK_H
