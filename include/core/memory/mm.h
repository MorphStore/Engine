/**********************************************************************************************
 * Copyright (C) 2019 by Johannes Pietrzyk                                                    *
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
 * @file mm.h
 * @brief Defines the memory managment interface ( abstract_memory_manager )
 *
 * @details Declares the pure virtual class abstract_memory_manager.
 *
 *
 * @author Johannes Pietrzyk
 *
 * @todo Overthink handle_error( )
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_H
#define MORPHSTORE_CORE_MEMORY_MM_H


#include "../utils/types.h"


#include <cstdlib>


namespace morphstore { namespace memory {

class abstract_memory_manager {
   public:
      virtual void * allocate( size_t ) = 0;
      virtual void * allocate( abstract_memory_manager * const, size_t ) = 0;
      virtual void deallocate( abstract_memory_manager * const, void * const ) = 0;
      virtual void deallocate( void * const ) = 0;
      virtual void handle_error( ) = 0;
};


} }

#endif //MORPHSTORE_CORE_MEMORY_MM_H
