#include <core/memory/mm_glob.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
int main( void ) {
    using namespace vectorlib;
    using namespace morphstore;

    //tests for avx2<v256<uint32_t>>:

    // { //test add
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,8
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test min
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,4,4,4,4
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test sub
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t subtracted = sub<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,...,7
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(subtracted, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test had, doesn't work yet
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t hadded = hadd<ps1, vector_base_t_granularity::value>::apply(sequence1); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(hadded, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test min
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,4,4,4,4
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test mul
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t multiplied = mul<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,2,4,6,8,10,12,14
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(multiplied, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test div, doesn't work yet
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(2,2); //2,4,6,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t divided = div<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(divided, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

   // { //test inv, not sure if output is correct
   //  using ps1 = avx2<v256<uint32_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
   //   std::cout<< "using avx2, 32bit" << std::endl;  
   //   vector_t inversed = inv<ps1, vector_base_t_granularity::value>::apply(sequence1); 
   //   uint32_t number1 = 0;
   //   for(int i=0; i < 8; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(inversed, i); 
   //       std::cout << number1 << std::endl; //4294967295, 4294967294, 4294967293, 4294967292, 4294967291, 4294967290
   //       //4294967299, 4294967298
   //   }
   //  }

    //tests for avx2<v256<uint16_t>>:

    // { //test add
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,15
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 16bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,16
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

   // { //test min
   //  using ps1 = avx2<v256<uint16_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,15
   //   vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
   //   std::cout<< "using avx2, 16bit" << std::endl;  
   //   vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,4,4,4,4,..
   //   uint16_t number1 = 0;
   //   for(int i=0; i < 16; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
   //       std::cout << number1 << std::endl;
   //   }
   //  }

    // { //test sub
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,16
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 16bit" << std::endl;  
    //  vector_t subtracted = sub<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,...,15
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(subtracted, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test mul
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,15
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using avx2, 16bit" << std::endl;  
    //  vector_t multiplied = mul<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,2,4,6,8,10,12,14,...
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(multiplied, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

   // { //test inv, not sure if output is correct
   //  using ps1 = avx2<v256<uint16_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,16
   //   std::cout<< "using avx2, 16bit" << std::endl;  
   //   vector_t inversed = inv<ps1, vector_base_t_granularity::value>::apply(sequence1); 
   //   uint16_t number1 = 0;
   //   for(int i=0; i < 16; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(inversed, i); 
   //       std::cout << number1 << std::endl; 
   //   }
   //  }


    //tests for avx2<v256<uint8_t>>:

    // { //test add
    // using ps1 = avx2<v256<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,31
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 8bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,32
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 32; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

   // { //test min
   //  using ps1 = avx2<v256<uint8_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,31
   //   vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
   //   std::cout<< "using avx2, 8bit" << std::endl;  
   //   vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,4,4,4,4,..
   //   uint8_t number1 = 0;
   //   for(int i=0; i < 32; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
   //       std::cout << unsigned(number1) << std::endl;
   //   }
   //  }

    // { //test sub
    // using ps1 = avx2<v256<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,32
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 8bit" << std::endl;  
    //  vector_t subtracted = sub<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,...,31
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 32; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(subtracted, i);
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

  // { //test inv, not sure if output is correct
  //   using ps1 = avx2<v256<uint8_t>>;
  //   IMPORT_VECTOR_BOILER_PLATE(ps1)

  //    vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,32
  //    std::cout<< "using avx2, 8bit" << std::endl;  
  //    vector_t inversed = inv<ps1, vector_base_t_granularity::value>::apply(sequence1); 
  //    uint8_t number1 = 0;
  //    for(int i=0; i < 32; i++){
  //        number1 = extract_value<ps1,vector_base_t_granularity::value>(inversed, i); 
  //        std::cout << unsigned(number1) << std::endl; 
  //    }
  //   }

}
 
