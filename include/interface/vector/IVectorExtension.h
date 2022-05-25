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


#ifndef MORPHSTORE_INCLUDE_INTERFACE_VECTOR_IVECTOREXTENSION_H
#define MORPHSTORE_INCLUDE_INTERFACE_VECTOR_IVECTOREXTENSION_H

namespace vectorlib {
    
    /// Forward declaration of class VectorExtension in <vector/vv/extension_virtual_vector.h>
    class VectorExtension;
    
    #ifdef USE_CPP20_CONCEPTS
    
        /// Interface for possible vector extensions
        template< typename TVectorExtension >
        concept IVectorExtension = std::is_base_of<VectorExtension, TVectorExtension>::value;
        
    #else
      
        #define IVectorExtension typename
      
    #endif
    
}
#endif //MORPHSTORE_INCLUDE_INTERFACE_VECTOR_IVECTOREXTENSION_H