#include "../../include/core/memory/mm_glob.h"
//#include <core/memory/mm_glob.h>

#include "../../include/core/morphing/format.h"
// #include "../../include/core/operators/scalar/agg_sum_uncompr.h"
// #include "../../include/core/operators/scalar/project_uncompr.h"
// #include "../../include/core/operators/scalar/select_uncompr.h"
#include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/operators/general_vectorized/project_uncompr.h>
#include <core/operators/general_vectorized/select_uncompr.h>

#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"

// #include "../../include/vector/scalar/extension_scalar.h"

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

// #include <core/morphing/format.h>
// #include <core/morphing/uncompr.h>

#include <functional>
#include <iostream>
#include <random>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************

    // #ifdef tally
    //   std::cout << "nice" << std::endl;
    //   output_tally();
    // #endif


    std::cout << "Base data generation started... ";
    std::cout.flush();

    const size_t countValues = 60;//100;//128/4; // generate 100 numbers
    const column<uncompr_f> * const myNumbers = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint8_t>(1, 20), //range between 1 and 50
            false //numbers are sorted
            // true
    );

    std::cout << "done." << std::endl;
    // print_columns(print_buffer_base::decimal, myNumbers, "myNumbers");//gives out the table

    // ************************************************************************
    // * Query execution
    // ************************************************************************

    // using ve = scalar<v64<uint64_t> >;
    using ve = sse<v128<uint8_t>>;
    // using ve = avx2<v256<uint8_t>>;
    // using ve = avx2<v256<uint64_t>>;
    // using ve = avx512<v512<uint64_t>>;


    std::cout << "Numbers of Elements: " << countValues << std::endl;
    std::cout << "Query execution started...\t";
    std::cout.flush();

    // #ifdef tally
    //   std::cout << std::endl<< "before i1" << std::endl;
    //   output_tally();
    //   reset_tally();
    // #endif

    // Positions fulfilling "myNumbers < 10"
    int sel_nr = 14;
    auto i1 = morphstore::select<
            less,
            ve,
            uncompr_f,
            uncompr_f
    >(myNumbers, sel_nr);

    #ifdef tally
      std::cout << std::endl << std::endl << "SELECT" << std::endl;
      output_tally();
      reset_tally();
    #endif
    // Data elements of "myNumbers" fulfilling "myNumbers < 10"
    // auto i2 = project<ve, uncompr_f>(myNumbers, i1);
    // std::cout << "done select...\t";
    // std::cout.flush();

    auto i2 = morphstore::project<
            ve,
            uncompr_f,
            uncompr_f,
            uncompr_f
      >(myNumbers, i1);
    // std::cout << "done project..." << std::endl << std::endl;

    #ifdef tally
      std::cout << std::endl << std::endl << "PROJECT" << std::endl;
      output_tally();
      reset_tally();
      std::cout << std::endl << std::endl;
    #endif
    // ************************************************************************
    // * Result output
    // ************************************************************************

    std::cout << "Element Count my NUmbers:\t"<< myNumbers->get_count_values()<<std::endl;
    std::cout << "Element Count Select:\t\t" << i1->get_count_values()<<std::endl;
    std::cout << "Element Count Project:\t\t" << i2->get_count_values()<<std::endl<<std::endl;

    // print_columns(print_buffer_base::decimal, myNumbers, "myNumbers");
    // print_columns(print_buffer_base::decimal, i1, "Idx myNumbers<10");
    // print_columns(print_buffer_base::decimal, i2, "myNumbers<10");

    uint8_t *mudata1 = myNumbers->get_data(), *mudata2 = i1->get_data(), *mudata3 = i2->get_data();
    uint32_t k = 0;

    std::cout << "IDX\tmyNumbers\tErgebnis IDX myNumbers <10\tErgebnis"<<std::endl;
    for(uint64_t i = 0; i < countValues; i++){
         std::cout <<i<<"\t" << (int)mudata1[i] << "\t";
         for(uint64_t w = 0; w <= i && w < i1->get_count_values();w++){
            if(mudata2[w] == i){
               std::cout <<"x";
               break;
            }
         }
         std::cout << "\t";
         if(k < i1->get_count_values()){
            std::cout << (uint64_t) mudata2[k] << "\t\t\t\t";
            std::cout << (uint64_t) mudata3[k] << "\t";
            if(mudata1[mudata2[k]] == mudata3[k]){
               if((int)mudata1[mudata2[k]] < sel_nr){
                  std::cout << "Fitting Number";
               }else{
                  std::cout << "Correct Number from that position. NOT FITTING";
               }
            }else{
               std::cout << "ERROR";
            }
         }
         std::cout << std::endl;
         k++;
         if(k%(8) == 0){
            std::cout << "-------------------\n";
         }

    }

    return 0;
}
