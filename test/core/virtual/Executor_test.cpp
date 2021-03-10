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
 

#include <virtual>
#include <utils>
#include <storage>
#include <operators>
#include <vectorExtensions>
#include <iomanip>

#include "compileTest.h"

using namespace std;

class op : public Executable {
  public:
    
    static column<uncompr_f> * apply(column<uncompr_f> * col1, column<uncompr_f> * col2, uint64_t * val1) {
        std::cout << "Called op::apply with Inputs: column<uncompr_f>*, column<uncompr_f>*, uint64_t * / "
        << "Output: column<uncompr_f>*" << std::endl;
        return reinterpret_cast<column<uncompr_f>*>(val1);
    }
    static column<uncompr_f> * apply(column<uncompr_f> * col1, column<uncompr_f> * col2) {
        std::cout << "Called op::apply with Inputs: column<uncompr_f>*, column<uncompr_f>* / Output: column<uncompr_f>*"
        << std::endl;
        return nullptr;
    }
    
    template<typename ... T>
    static uint64_t apply(PartitionedColumn<T...> * colIn){
        std::cout << "Called op::apply with 1 PartitionedColumn<...> as input and 1 uint64_t as output." << std::endl;
        return 0;
    }
    template<typename ... T>
    static uint64_t apply(column<T...> * colIn){
        std::cout << "Called op::apply with 1 column<...> as input and 1 uint64_t as output." << std::endl;
        return 0;
    }
    
    static uint64_t apply(uint64_t){
        std::cout << "Called op::apply with 1 uint64_t as input and 1 uint64_t as output." << std::endl;
        return 0;
    }
    
    static uint64_t apply(uint64_t*){
        std::cout << "Called op::apply with 1 uint64_t* as input and 1 uint64_t as output." << std::endl;
        return 0;
    }
};

class op2 : public Executable {
  public:
    using inputType = InputTypes<uint64_t*>;
    using outputType = OutputTypes<uint64_t>;
    
    static uint64_t apply(uint64_t* in){
        std::cout << "Called op::apply with 1 uint64_t* as input and 1 uint64_t as output. Value = " << *in << std::endl;
        return 0;
    }
};

class op_2_columns_in : public Executable {
  public:
    static const column<uncompr_f> * apply(const column<uncompr_f> * col1, const column<uncompr_f> * col2) {
        std::cout << "Called op_2_columns_in::apply with Inputs: const column<uncompr_f>*, const column<uncompr_f>* / Output: const column<uncompr_f>*"
        << std::endl;
        return nullptr;
    }
};

void print_column(const column<uncompr_f> * col){
    size_t size = col->get_count_values();
    uint64_t * data = col->get_data();
    
    for(size_t i = 0; i < size; ++i){
        std::cout << "Line " << i << ": " << data[i] << std::endl;
    }
    
}

template<typename first, typename ... last>
size_t get_max_size(first f, last ... l){
    if constexpr(sizeof...(last) > 1) {
        return std::max(f->get_count_values(), get_max_size(l...));
    } else {
        return f->get_count_values();
    }
}

void print_element(size_t index, const column<uncompr_f> * col, size_t width){
    std::cout << std::setw(width);
    if(index < col->get_count_values()) {
        std::cout << ((uint64_t*)col->get_data())[index] << " | ";
    } else {
        std::cout << "NaN" << " | ";
    }
}

template<typename...TCols>
void print_columns(TCols...cols){
    for(size_t i = 0; i < get_max_size(cols...); ++i){
        (print_element(i, cols, 5), ...);
        std::cout << std::endl;
    }
}


int main() {
    
    /// 2 GB of data
//    const size_t countValues = 1024 * 1024 / sizeof(uint64_t) * 2 * 1024;
    const size_t countValues = 8 * 10;
    
    uint64_t threadCnt = 4;
    
    column<uncompr_f> * const col1 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                8
            )
          )
      );
    column<uncompr_f> * const col2 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                7
            )
          )
      );
    
//
//    std::cout << "Input column 1:" << std::endl;
//    print_column(col1);
//    std::cout << "Input column 2:" << std::endl;
//    print_column(col2);


//    auto pcol1 = new PartitionedColumn< uncompr_f, uint64_t, LogicalPartitioner<column<uncompr_f>, uint64_t, 64> >(col1, threadCnt);
//    auto pcol1 = new PartitionedColumn< uncompr_f, uint64_t, LogicalPartitioner<column<uncompr_f>, uint64_t, 64> >(col1, threadCnt);
//    auto pcol2 = new PartitionedColumn< uncompr_f, uint64_t, LogicalPartitioner<column<uncompr_f>, uint64_t, 64> >(col2, threadCnt);
    //// @todo === New Implementation ===
    auto pcol1 = new PartitionedColumn< LogicalPartitioner, std::remove_pointer<decltype(col1)>::type::format > ( col1, threadCnt );
    auto pcol2 = new PartitionedColumn< LogicalPartitioner, std::remove_pointer<decltype(col1)>::type::format > ( col2, threadCnt );
    
//    VirtualArray<uint64_t> val1(0x010);
    uint64_t * val1 = new uint64_t(0x010);
    
    std::tuple<uint64_t> t;
    
//    Executor<ConcurrentType::STD_THREADS, op, column<uncompr_f>, InputTypes<decltype(*col1), decltype(*col2), decltype(val1)>>::apply()

    std::vector<   OutputTypes< column<uncompr_f> * >*   >* result
      = Executor<
            ConcurrentType::STD_THREADS,
            op,
            OutputTypes<column<uncompr_f>*>,
            InputTypes<decltype(pcol1), decltype(pcol2), decltype(val1)>
//            InputTypes<decltype(pcol1), decltype(pcol2) >
//            InputTypes<decltype(pcol1)>
        >::apply( 4, pcol1, pcol2, val1 );
    
    /// In  : uint64_t
    /// Out : uint64_t#
//    std::vector<OutputTypes<uint64_t>*> * result_1
//    = Executor<
//        ConcurrentType::STD_THREADS,
//        op,
//        OutputTypes<uint64_t>,
//        InputTypes<uint64_t>
//      >::apply(4,124176);
    
    
    /// In  : uint64_t
    /// Out : uint64_t
//    uint64_t * val_2 = new uint64_t(25);
//    std::vector<OutputTypes<uint64_t>*> * result_2
//    = Executor<
//        ConcurrentType::STD_THREADS,
//        op,
//        OutputTypes<uint64_t>,
//        InputTypes<uint64_t*>
//      >::apply(4,val_2);
    
    /// In  : PartitionedColumn*
    /// Out : uint64_t
//    std::vector<OutputTypes<uint64_t>*> * result_3
//    = Executor<
//        ConcurrentType::STD_THREADS,
//        op,
//        OutputTypes<uint64_t>,
//        InputTypes<decltype(pcol1)>
//      >::apply(4,pcol1);
    
    uint64_t * val_3 = new uint64_t(23);
    std::vector<OutputTypes<uint64_t>*> * result_4
    = Executor<
        ConcurrentType::STD_THREADS,
        op2,
        op2::outputType,
        op2::inputType
//        ,OutputTypes<uint64_t>
//        ,InputTypes<uint64_t*>
      >::apply(4, val_3);
    
    /// @todo: ...
//    uint64_t * val_5 = new uint64_t(42);
//    std::vector<typename op2::outputType*> * result_5
//    = AnotherExecutor<
//      ConcurrentType::STD_THREADS,
//      op2
//      >::apply(4, val_5);


    using ps = vectorlib::scalar<vectorlib::v64<uint64_t>>;

    /// Execution
    std::vector<   OutputTypes< const column<uncompr_f> * >*   >* result_calc_add
      = Executor<
            ConcurrentType::STD_THREADS,
            calc_binary_t<std::plus, ps, uncompr_f, uncompr_f, uncompr_f>,
//            op_2_columns_in,
            OutputTypes<const column<uncompr_f>*>,
            InputTypes<decltype(pcol1), decltype(pcol2)>
//            InputTypes<decltype(pcol1), decltype(pcol2) >
//            InputTypes<decltype(pcol1)>
        >::apply( threadCnt, pcol1, pcol2);
    
    /// Consolidation
//    PartitionedColumn<uncompr_f, uint64_t, PhysicalPartitioner<column<uncompr_f>, uint64_t>> presult;
    PartitionedColumn< PhysicalPartitioner<column<uncompr_f>, uint64_t>, uncompr_f, uint64_t > presult;
    for(auto& outTuple : *result_calc_add){
        presult.addPartition(get<0>(*outTuple)->deconst());
    }
    
    /// test
    const column<uncompr_f> * const_col = new const column<uncompr_f>(sizeof(uint64_t) * 8);
    {
        uint64_t * data = const_col->get_data();
        for (uint64_t i = 0; i < 8; ++ i) {
            data[i] = i;
        }
    }
    column<uncompr_f> * col = const_col->deconst();
    
    std::cout << typestr(const_col) << std::endl;
    std::cout << typestr(col) << std::endl;
    
    {
        uint64_t * data = const_col->get_data();
        for (uint64_t i = 0; i < 8; ++ i) {
            std::cout << "[" << i << "]" << data[i] << std::endl;
        }
    }
    
    
    {
        uint64_t * data = col->get_data();
        for (uint64_t i = 0; i < 8; ++ i) {
            std::cout << "[" << i << "]" << data[i] << std::endl;
        }
    }
    uint64_t part = 0;
    
//    std::cout << "Input column 1:" << std::endl;
//    for(uint64_t i = 0; i < threadCnt; ++i){
//        std::cout << "Part" << part++ << ":" << std::endl;
//        column<uncompr_f> * col = (*pcol1)[i];
//        print_column(col);
//    }
//    std::cout << "Input column 2:" << std::endl;
//    for(uint64_t i = 0; i < threadCnt; ++i){
//        std::cout << "Part" << part++ << ":" << std::endl;
//        column<uncompr_f> * col = (*pcol2)[i];
//        print_column(col);
//    }
//
//    std::cout << "Output column:" << std::endl;
//    for(auto out : *result_calc_add) {
//        std::cout << "Part" << part++ << ":" << std::endl;
//        auto column = std::get<0>(*out);
//        print_column(column);
//    }
    
    std::cout << "Output column:" << std::endl;
    for(uint64_t i = 0; i < threadCnt; ++i){
        cout << "Part" << part++ << ":" << std::endl;
        cout << setw(5) << "IN1" << " | " << setw(5) << "IN2" << " | " << setw(5) << "OUT" << " | " << endl;
        auto result = std::get<0>(*result_calc_add->at(i));
        column<uncompr_f> * col1 = (*pcol1)[i];
        column<uncompr_f> * col2 = (*pcol2)[i];
        column<uncompr_f> * pcol = presult[i];
        print_columns(col1, col2, result, pcol);
    }
    
    return 0;
}
