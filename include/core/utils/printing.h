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
 * @file printing.h
 * @brief Some utilities for printing columns.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_PRINTING_H
#define MORPHSTORE_CORE_UTILS_PRINTING_H

#include <core/storage/column.h>
#include <core/utils/math.h>
#include <core/utils/basic_types.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace morphstore {

/**
 * Prints the binary representation of the given number of lower bytes of the
 * given value to the given stream.
 * @param val
 * @param countBytes
 * @param os
 * @param zeroChar
 * @param oneChar
 */
template< typename uintX_t >
void print_binary(
        std::ostream & os,
        uintX_t val,
        size_t countBytes,
        char zeroChar = '.',
        char oneChar = 'I'
) {
    for( signed i = countBytes * bitsPerByte - 1; i >= 0; i-- ) {
        os << ( ( ( val >> i ) & 1 ) ? oneChar : zeroChar );
        if( !( i % 8 ) && ( i > 0 ) )
            os << "-";
    }
}

/**
 * Writes the given character to the given stream the given number of times.
 * @param os
 * @param c
 * @param n
 */
void repeat_char( std::ostream & os, char c, size_t n ) {
    os << std::setw( n ) << std::setfill( c ) << "";
}

/**
 * Bases for printing columns respectively buffers.
 */
enum class print_buffer_base {
    binary,
    decimal,
    hexadecimal
};

/**
 * Struct containing all information needed to print a buffer.
 */
struct print_buffer_info {
    const std::string & m_Title;
    const uint8_t * m_Buffer;
    const size_t m_SizeByte;
    
    print_buffer_info(
            const std::string & p_Title,
            const uint8_t * p_Buffer,
            size_t p_SizeByte
    ) :
        m_Title( p_Title ),
        m_Buffer( p_Buffer ),
        m_SizeByte( p_SizeByte )
    {
        //
    }
    
    template< class F >
    print_buffer_info(
            const std::string & p_Title,
            const column< F > * p_Col
    ) :
        m_Title( p_Title ),
        m_Buffer( p_Col->get_data( ) ),
        m_SizeByte( p_Col->get_size_used_byte( ) )
    {
        //
    }
    
    // TODO make the copy constructor unnecessary?
    print_buffer_info( print_buffer_info const & ) = default;
    print_buffer_info( print_buffer_info && ) = default;
    print_buffer_info & operator=( print_buffer_info const & ) = delete;
    print_buffer_info & operator=( print_buffer_info && ) = default;
};

/**
 * Prints the given buffers to std::cout side by side in the specified
 * representation. One uintX_t-word per line is output for each buffer. This is
 * a low-level function, consider using print_columns instead.
 * @param p_Base
 * @param p_Infos
 */
// TODO support individual base and width for each buffer
// TODO support types wider than uint64_t: __m128i, __m256i, __m512i
template< typename uintX_t = uint64_t >
void print_buffers(
        print_buffer_base p_Base,
        const std::vector< print_buffer_info > & p_Infos
) {
    // Save cout's format flags.
    // TODO ensure that the flags are restored even if an exception is thrown
    std::ios::fmtflags flagsBefore = std::cout.flags( );
    
    // Determine the width of one output column.
    size_t colW;
    size_t digitsPerByte;
    switch( p_Base ) {
        case print_buffer_base::binary:
            digitsPerByte = bitsPerByte;
            // Note that bytes are separated by a dash character in the output.
            colW = sizeof( uintX_t ) * ( digitsPerByte + 1 ) - 1;
            break;
        case print_buffer_base::hexadecimal:
            digitsPerByte = 2;
            colW = sizeof( uintX_t ) * digitsPerByte;
            break;
        case print_buffer_base::decimal:
            colW = std::numeric_limits< uintX_t >::digits10 + 1;
            digitsPerByte = 0; // this is unused for decimal output
            break;
        default:
            throw std::runtime_error( "unsupported print_buffer_base" );
    }
    
    // Determine the number of buffers.
    const size_t countBufs = p_Infos.size( );

    // Determine the maximum buffer size.
    size_t maxSizeByte = 0;
    for( unsigned bufIdx = 0; bufIdx < countBufs; bufIdx++ ) {
        const size_t sizeByte = p_Infos[ bufIdx ].m_SizeByte;
        if( sizeByte > maxSizeByte )
            maxSizeByte = sizeByte;
    }
    // 1+ due to the extra line saying that it is the end.
    const size_t maxCountLines = 1 + round_up_div(
            maxSizeByte,
            sizeof( uintX_t )
    );
    
    // Some config.
    // The separator between columns in the output.
    const std::string colSpaceStr = std::string( "  " );
    // The character used for digits in bytes beyond a buffer's end.
    const char overflowChar = '#';
    // The character used for leading zero digits.
    const char leadingZeroChar = '.';
    // TODO use this also in print_binary
    // The character used to visually separate bytes in the binary output.
    const char binByteSepChar = '-';
    // The title of the index column.
    const std::string indexWord( "index" );
    // The width of the index column (must suffice for the title and the
    // highest index value).
    const size_t indexColW = std::max(
            static_cast< size_t >( ceil( log10( maxCountLines ) ) ),
            indexWord.length( )
    );
    // The word that is printed in the first line after a buffer's end.
    const std::string endWord( "(end)" );
    // The length of the end-word.
    const size_t lenEndWord = endWord.length( );
    // For each buffer: the number of extra spaces required if the buffer's
    // title or the end-word is wider than the values in the respective base.
    std::vector< size_t > extraSpaces;
    for( unsigned bufIdx = 0; bufIdx < countBufs; bufIdx++ )
        extraSpaces.push_back( std::max(
                static_cast< int >( std::max(
                        lenEndWord,
                        p_Infos[ bufIdx ].m_Title.length( )
                ) ) - static_cast< int >( colW ),
                0
        ) );
    
    // Print the headline.
    std::cout << std::setw( indexColW ) << indexWord;
    std::cout << std::setfill( ' ' ) << std::left;
    for( unsigned bufIdx = 0; bufIdx < countBufs; bufIdx++ )
        std::cout 
                << colSpaceStr << std::setw( colW + extraSpaces[ bufIdx ] )
                << p_Infos[ bufIdx ].m_Title;
    std::cout << std::endl;

    // Print the buffers.
    std::cout << std::noshowbase;
    for( unsigned lineIdx = 0; lineIdx < maxCountLines; lineIdx++ ) {
        std::cout
                << std::setw( indexColW ) << std::setfill( ' ' ) << std::right
                << std::dec << lineIdx;
        const size_t posByte = lineIdx * sizeof( uintX_t );
        const size_t posByteNext = posByte + sizeof( uintX_t );
        const size_t posBytePrev = posByte - sizeof( uintX_t );
        for( unsigned bufIdx = 0; bufIdx < countBufs; bufIdx++ ) {
            const size_t sizeByte = p_Infos[ bufIdx ].m_SizeByte;
            std::cout << colSpaceStr;
            if( posByteNext <= sizeByte ) {
                // We have not reached this buffer's end yet.
                // We can still read a complete uintX_t-word from this buffer.
                
                std::cout << std::setfill( leadingZeroChar ) << std::right;
                
                // The value is casted here, because for uint8_t the output
                // would be rubbish otherwise.
                const uint64_t value = static_cast< uint64_t >(
                        *reinterpret_cast< const uintX_t * >(
                                p_Infos[ bufIdx ].m_Buffer + posByte
                        )
                );
                
                // Print the uintX_t-word in the specified base.
                switch( p_Base ) {
                    case print_buffer_base::binary:
                        print_binary( std::cout, value, sizeof( uintX_t ) );
                        break;
                    case print_buffer_base::decimal:
                        std::cout << std::setw( colW ) << std::dec << value;
                        break;
                    case print_buffer_base::hexadecimal:
                        std::cout << std::setw( colW ) << std::hex << value;
                        break;
                    default:
                        throw std::runtime_error("unsupported print_buffer_base");
                }
                
                // Some extra space if this buffer's title or the end-word is
                // wider than the value itself.
                repeat_char( std::cout, ' ', extraSpaces[ bufIdx ] );
            }
            else if( posByte < sizeByte && posByteNext > sizeByte ) {
                // We are reaching this buffer's end now.
                // We can only read a part of a uintX_t word from this buffer.
                
                // Determine how many bytes of this buffer can still be read
                // and how many bytes are overflow bytes, i.e., behind the end
                // of this buffer.
                const size_t restBytes = sizeByte % sizeof( uintX_t );
                const size_t overflowBytes = sizeof( uintX_t ) - restBytes;
                
                // Load the value bytewise in order to prevent a reading buffer
                // overflow. The value is a uint64_t here, because for uint8_t
                // the output would be rubbish otherwise.
                uint64_t value = 0;
                for( unsigned byteIdx = 0; byteIdx < restBytes; byteIdx++ )
                    value |=
                            p_Infos[ bufIdx ].m_Buffer[ posByte + byteIdx ]
                            << ( bitsPerByte * byteIdx );
                
                // Print some overflow characters followed by the available
                // part of the uintX_t-word in the specified base.
                switch( p_Base ) {
                    case print_buffer_base::binary:
                        for( unsigned i = 0; i < overflowBytes; i++ ) {
                            repeat_char( std::cout, overflowChar, digitsPerByte );
                            std::cout << binByteSepChar;
                        }
                        print_binary( std::cout, value, restBytes );
                        break;
                    case print_buffer_base::decimal: {
                        unsigned restDigits = static_cast< unsigned >(
                                ceil( log10(
                                        std::numeric_limits< uintX_t >::max( )
                                        >> ( bitsPerByte * overflowBytes )
                                ) )
                        );
                        repeat_char( std::cout, overflowChar, colW - restDigits );
                        std::cout
                                << std::setfill( leadingZeroChar )
                                << std::setw( restDigits )
                                << std::dec << value;
                        break;
                    }
                    case print_buffer_base::hexadecimal:
                        repeat_char(
                                std::cout,
                                overflowChar,
                                overflowBytes * digitsPerByte
                        );
                        std::cout
                                << std::setfill( leadingZeroChar )
                                << std::setw( restBytes * digitsPerByte )
                                << std::hex << value;
                        break;
                    default:
                        throw std::runtime_error( "unsupported print_buffer_base" );
                }
                
                repeat_char( std::cout, ' ', extraSpaces[ bufIdx ] );
            }
            else if( posBytePrev < sizeByte && posByte >= sizeByte ) {
                // This is the first line after this buffer's end.
                
                std::cout
                        << std::setw( colW + extraSpaces[ bufIdx ] )
                        << std::setfill( ' ' ) << std::left << endWord;
            }
            else {
                // More lines after this buffer's end (until the ends of all
                // other buffers have been reached).
                
                repeat_char( std::cout, '_', colW );
                repeat_char( std::cout, ' ', extraSpaces[ bufIdx ]);
            }

        }
        std::cout << std::endl;
    }
    
    // Restore cout's format flags.
    std::cout.flags( flagsBefore );
}

// TODO it is very ugly to have one function per number of columns to print...

/**
 * Print one column in the specified base, one uintX_t-word per line.
 * @param p_Base
 * @param p_Col1
 * @param p_Title1
 */
template< typename uintX_t = uint64_t, class F1 >
void print_columns(
        print_buffer_base p_Base,
        const column< F1 > * p_Col1,
        const std::string & p_Title1 = "column 1"
) {
    print_buffers< uintX_t >(
            p_Base,
            {
                print_buffer_info( p_Title1, p_Col1 )
            }
    );
}

/**
 * Prints two columns in the specified base, one uintX_t-word per line.
 * @param p_Base
 * @param p_Col1
 * @param p_Col2
 * @param p_Title1
 * @param p_Title2
 */
template< typename uintX_t = uint64_t, class F1, class F2 >
void print_columns(
        print_buffer_base p_Base,
        const column< F1 > * p_Col1,
        const column< F2 > * p_Col2,
        const std::string & p_Title1 = "column 1",
        const std::string & p_Title2 = "column 2"
) {
    print_buffers< uintX_t >(
            p_Base,
            {
                print_buffer_info( p_Title1, p_Col1 ),
                print_buffer_info( p_Title2, p_Col2 )
            }
    );
}

/**
 * Prints three columns in the specified base, one uintX_t-word per line.
 * @param p_Base
 * @param p_Col1
 * @param p_Col2
 * @param p_Col3
 * @param p_Title1
 * @param p_Title2
 * @param p_Title3
 */
template< typename uintX_t = uint64_t, class F1, class F2, class F3 >
void print_columns(
        print_buffer_base p_Base,
        const column< F1 > * p_Col1,
        const column< F2 > * p_Col2,
        const column< F3 > * p_Col3,
        const std::string & p_Title1 = "column 1",
        const std::string & p_Title2 = "column 2",
        const std::string & p_Title3 = "column 3"
) {
    print_buffers< uintX_t >(
            p_Base,
            {
                print_buffer_info( p_Title1, p_Col1 ),
                print_buffer_info( p_Title2, p_Col2 ),
                print_buffer_info( p_Title3, p_Col3 )
            }
    );
}

/**
 * Prints four columns in the specified base, one uintX_t-word per line.
 * @param p_Base
 * @param p_Col1
 * @param p_Col2
 * @param p_Col3
 * @param p_Col4
 * @param p_Title1
 * @param p_Title2
 * @param p_Title3
 * @param p_Title4
 */
template< typename uintX_t = uint64_t, class F1, class F2, class F3, class F4 >
void print_columns(
        print_buffer_base p_Base,
        const column< F1 > * p_Col1,
        const column< F2 > * p_Col2,
        const column< F3 > * p_Col3,
        const column< F4 > * p_Col4,
        const std::string & p_Title1 = "column 1",
        const std::string & p_Title2 = "column 2",
        const std::string & p_Title3 = "column 3",
        const std::string & p_Title4 = "column 4"
) {
    print_buffers< uintX_t >(
            p_Base,
            {
                print_buffer_info( p_Title1, p_Col1 ),
                print_buffer_info( p_Title2, p_Col2 ),
                print_buffer_info( p_Title3, p_Col3 ),
                print_buffer_info( p_Title4, p_Col4 )
            }
    );
}

/**
 * Prints five columns in the specified base, one uintX_t-word per line.
 * @param p_Base
 * @param p_Col1
 * @param p_Col2
 * @param p_Col3
 * @param p_Col4
 * @param p_Col5
 * @param p_Title1
 * @param p_Title2
 * @param p_Title3
 * @param p_Title4
 * @param p_Title5
 */
template< typename uintX_t = uint64_t, class F1, class F2, class F3, class F4, class F5 >
void print_columns(
        print_buffer_base p_Base,
        const column< F1 > * p_Col1,
        const column< F2 > * p_Col2,
        const column< F3 > * p_Col3,
        const column< F4 > * p_Col4,
        const column< F5 > * p_Col5,
        const std::string & p_Title1 = "column 1",
        const std::string & p_Title2 = "column 2",
        const std::string & p_Title3 = "column 3",
        const std::string & p_Title4 = "column 4",
        const std::string & p_Title5 = "column 5"
) {
    print_buffers< uintX_t >(
            p_Base,
            {
                print_buffer_info( p_Title1, p_Col1 ),
                print_buffer_info( p_Title2, p_Col2 ),
                print_buffer_info( p_Title3, p_Col3 ),
                print_buffer_info( p_Title4, p_Col4 ),
                print_buffer_info( p_Title5, p_Col5 )
            }
    );
}

/**
 * Prints six columns in the specified base, one uintX_t-word per line.
 * @param p_Base
 * @param p_Col1
 * @param p_Col2
 * @param p_Col3
 * @param p_Col4
 * @param p_Col5
 * @param p_Col6
 * @param p_Title1
 * @param p_Title2
 * @param p_Title3
 * @param p_Title4
 * @param p_Title5
 * @param p_Title6
 */
template< typename uintX_t = uint64_t, class F1, class F2, class F3, class F4, class F5, class F6 >
void print_columns(
        print_buffer_base p_Base,
        const column< F1 > * p_Col1,
        const column< F2 > * p_Col2,
        const column< F3 > * p_Col3,
        const column< F4 > * p_Col4,
        const column< F5 > * p_Col5,
        const column< F6 > * p_Col6,
        const std::string & p_Title1 = "column 1",
        const std::string & p_Title2 = "column 2",
        const std::string & p_Title3 = "column 3",
        const std::string & p_Title4 = "column 4",
        const std::string & p_Title5 = "column 5",
        const std::string & p_Title6 = "column 6"
) {
    print_buffers< uintX_t >(
            p_Base,
            {
                print_buffer_info( p_Title1, p_Col1 ),
                print_buffer_info( p_Title2, p_Col2 ),
                print_buffer_info( p_Title3, p_Col3 ),
                print_buffer_info( p_Title4, p_Col4 ),
                print_buffer_info( p_Title5, p_Col5 ),
                print_buffer_info( p_Title6, p_Col6 )
            }
    );
}

}
#endif //MORPHSTORE_CORE_UTILS_PRINTING_H
