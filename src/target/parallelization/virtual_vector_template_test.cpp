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
 
#include <iostream>

#include <core/utils/logger.h>
/// primitives
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

using namespace morphstore;
using namespace vectorlib;
using namespace virtuallib;

int main() {

    using baseVectorExtension = avx512<v512<int32_t>>;
    
    using vv_core     = vv<VectorBuilder<ConcurrentType::OPEN_MP, 1, seq, 1, baseVectorExtension> >;
    using vv_batch    = vv<VectorBuilder<ConcurrentType::OPEN_MP, 1, seq, 1, vv_core            > >;
    using vv_operator = vv<VectorBuilder<ConcurrentType::OPEN_MP, 1, seq, 1, vv_batch           > >;

    
    std::cout << "Base Type [Base]:     " << typeid(baseVectorExtension::base_t).name() << std::endl;
    std::cout << "Base Type [Core]:     " << typeid(vv_core::base_t).name()             << std::endl;
    std::cout << "Base Type [Batch]:    " << typeid(vv_batch::base_t).name()            << std::endl;
    std::cout << "Base Type [Operator]: " << typeid(vv_operator::base_t).name()         << std::endl;
    
    
}
