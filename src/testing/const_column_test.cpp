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
 
#include <storage>
#include <utils>
#include <printing>


int main(){
    using namespace morphstore;
    using namespace std;

    const column<uncompr_f> * const baseCol1 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            generate_with_distr(
                10,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                8
            )));
    
    cout << "baseCol1 |" << endl;
    print_columns<8>(baseCol1);
    
//    ;/*
    uint64_t * data = baseCol1->get_data();
    
    cout << "try to manipulate const column data, which shouldn't be possible, right?" << endl;
    for(uint64_t i = 0; i < 10; ++i){
        data[i] = i;
    }
    
    
    cout << "baseCol1 |" << endl;
    print_columns<8>(baseCol1);
    /**/
}
