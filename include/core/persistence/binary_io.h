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
 * @file binary_io.h
 * @brief Loading/storing a column from/to a simple binary file.
 * @author Patrick Damme
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_PERSISTENCE_BINARY_IO_H
#define MORPHSTORE_CORE_PERSISTENCE_BINARY_IO_H

#include "../memory/mm_glob.h"
#include "../storage/column.h"
#include "../storage/column_helper.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace morphstore { namespace persistence {
   
template< typename T >
class binary_io {
    public:
        
        static const storage::column< T > * load( const std::string & p_Filename ) {
            using namespace std;
            using namespace storage;

            ifstream ifs( p_Filename, ios::in | ios::binary );

            if( ifs.good( ) ) {
                size_t countValues;
                size_t sizeByte;

                ifs.read( reinterpret_cast< char * >( & countValues ), sizeof( uint64_t ) );
                ifs.read( reinterpret_cast< char * >( & sizeByte )   , sizeof( uint64_t ) );
                if( !ifs.good( ) )
                    throw runtime_error("could not read the column meta data");

                column< T > * col = column< T >::createPerpetualColumn( sizeByte );

                ifs.read( reinterpret_cast< char * >( col->data( ) ), sizeByte );
                if( !ifs.good( ) )
                    throw runtime_error("could not read the column data");

                col->count_values( countValues );
                col->size_used_byte( sizeByte );

                return col;
            }
            else
                throw runtime_error("could not open the file for reading");
        }

        static void store( const storage::column< T > * p_Column, const std::string & p_FileName ) {
            using namespace std;

            ofstream ofs( p_FileName, ios::out | ios::binary );
            
            if( ofs.good( ) ) {
                const size_t countValues  = p_Column->count_values( );
                const size_t sizeUsedByte = p_Column->size_used_byte( );

                ofs.write( reinterpret_cast< const char * >( & countValues )    , sizeof( uint64_t ) );
                ofs.write( reinterpret_cast< const char * >( & sizeUsedByte )   , sizeof( uint64_t ) );
                ofs.write( reinterpret_cast< const char * >( p_Column->data( ) ), sizeUsedByte );
                if( !ofs.good( ) )
                    throw runtime_error("could write the column meta data and data");
            }
            else
                throw runtime_error("could open the file for writing");
        }
};

} }

#endif /* MORPHSTORE_CORE_PERSISTENCE_BINARY_IO_H */
