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


#ifndef MORPHSTORE_INCLUDE_INTERFACE_CORE_IOPERATORINPUT_H
#define MORPHSTORE_INCLUDE_INTERFACE_CORE_IOPERATORINPUT_H

namespace morphstore {
    #ifdef USE_CONCEPTS
        /// Interface for possible input types [executables and arithmetic types]
        template< typename TDataType >
        concept IOperatorInput =
          std::is_base_of<Storage, TDataType>::value
          or std::is_arithmetic<TDataType>::value
          or morphstore::is_storage_ptr<TDataType>::value
//          or std::is_arithmetic<typename std::remove_pointer<TDataType>::type>::value;
          or morphstore::is_arithmetic_ptr<TDataType>::value
      ;
    #else
      #define IOperatorInput typename
    #endif
}
#endif //MORPHSTORE_INCLUDE_INTERFACE_CORE_IOPERATORINPUT_H
