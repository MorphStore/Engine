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
 * @file property_type.h
 * @brief variant of supported data types as a property 
 * @todo Move into dedicated sub-folder (when different property mappings exists)
*/

#ifndef MORPHSTORE_PROPERTY_TYPE_H
#define MORPHSTORE_PROPERTY_TYPE_H

#include <variant>
#include <iostream>

namespace morphstore{
    // only to used if properties are stored per node or triple store
    // TODO: handle date and datetime properties and maybe text
    using property_type = std::variant<std::string, uint64_t>;

    struct PropertyValueVisitor {
            void operator()(const std::string &s) const {
                std::cout << "(string) " << s;
            }
            void operator()(uint64_t i) const
            {
                std::cout << "(uint_64t) " << i;
            }
    };

}


#endif //MORPHSTORE_PROPERTY_TYPE_H