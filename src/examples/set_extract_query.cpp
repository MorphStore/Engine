#include <core/memory/mm_glob.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
int main( void ) {
    using namespace vectorlib;
    using namespace morphstore;
    {
    using ps= sse<v128<uint16_t>>;
    IMPORT_VECTOR_BOILER_PLATE(ps)

    vector_t sequence = set_sequence<ps, vector_base_t_granularity::value>(0,1);
    std::cout<< "using sse, 16bit" << std::endl;
    uint16_t number = 0;
    for(int i=0; i < 8; i++){
        number = extract_value<ps,vector_base_t_granularity::value>(sequence, i);
        std::cout << number << std::endl;
    }
    }
    
    {
    using ps2= sse<v128<uint8_t>>;
    IMPORT_VECTOR_BOILER_PLATE(ps2)
    
    vector_t sequence2 = set_sequence<ps2, vector_base_t_granularity::value>(2,2);
    std::cout<< "using sse, 8bit" << std::endl;
    uint8_t number2 = 0;
    for(int i=0; i <  16; i++){
        number2 = extract_value<ps2,vector_base_t_granularity::value>(sequence2, i);
        std::cout << unsigned(number2) << std::endl;               
    }
    }
    
    {
    using ps3 = avx2<v256<uint16_t>>;
    IMPORT_VECTOR_BOILER_PLATE(ps3)

    vector_t sequence3 = set_sequence<ps3, vector_base_t_granularity::value>(0,1);
    std::cout<< "using avx2, 16bit" << std::endl;    
    uint16_t number3 = 0;
    for(int i=0; i < 16; i++){
        number3 = extract_value<ps3,vector_base_t_granularity::value>(sequence3, i);
        std::cout << number3 << std::endl;
    }
    }
    
    {
    using ps4 = avx2<v256<uint8_t>>;
    IMPORT_VECTOR_BOILER_PLATE(ps4)

    vector_t sequence4 = set_sequence<ps4, vector_base_t_granularity::value>(0,1);
    std::cout<< "using avx2, 8bit" << std::endl;    
    uint8_t number4 = 0;
    for(int i=0; i < 32; i++){
        number4 = extract_value<ps4,vector_base_t_granularity::value>(sequence4, i);
        std::cout << unsigned(number4) << std::endl;
    }     
    }
    
}
