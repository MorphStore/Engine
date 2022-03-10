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
 * @file binary_io.h
 * @brief Loading/storing a column from/to a simple binary file.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_PERSISTENCE_BINARY_IO_H
#define MORPHSTORE_CORE_PERSISTENCE_BINARY_IO_H

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_helper.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace morphstore {
   
template< class F >
class binary_io {
    public:
        
        static const column< F > * load( const std::string & p_Filename ) {
            std::ifstream ifs( p_Filename, std::ios::in | std::ios::binary );

            if( ifs.good( ) ) {
                size_t countValues;
                size_t sizeByte;

                ifs.read( reinterpret_cast< char * >( & countValues ), sizeof( uint64_t ) );
                ifs.read( reinterpret_cast< char * >( & sizeByte )   , sizeof( uint64_t ) );
                if( !ifs.good( ) )
                    throw std::runtime_error("could not read the column meta data");

                column< F > * col = column<F>::create_global_column(sizeByte);

                ifs.read( col->get_data(), sizeByte );
                if( !ifs.good( ) )
                    throw std::runtime_error("could not read the column data");

                col->set_count_values( countValues );
                col->set_size_used_byte( sizeByte );

                return col;
            }
            else
                throw std::runtime_error("could not open the file for reading");
        }

        static void store( const column< F > * p_Column, const std::string & p_FileName ) {
            std::ofstream ofs( p_FileName, std::ios::out | std::ios::binary );
            
            if( ofs.good( ) ) {
                const size_t countValues  = p_Column->get_count_values( );
                const size_t sizeUsedByte = p_Column->get_size_used_byte( );

                ofs.write( reinterpret_cast< const char * >( & countValues )    , sizeof( uint64_t ) );
                ofs.write( reinterpret_cast< const char * >( & sizeUsedByte )   , sizeof( uint64_t ) );
                ofs.write( p_Column->get_data( ), sizeUsedByte );
                if( !ofs.good( ) )
                    throw std::runtime_error("could write the column meta data and data");
            }
            else
                throw std::runtime_error("could open the file for writing");
        }
};

}

#endif /* MORPHSTORE_CORE_PERSISTENCE_BINARY_IO_H */
