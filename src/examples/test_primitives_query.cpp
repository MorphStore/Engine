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

    // { //test hadd
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  base_t hadded = vectorlib::hadd<ps1, vector_base_t_granularity::value>::apply(sequence1); 
    //  std::cout << unsigned(hadded) << std::endl;
    // }

    // { //test hadd
    // using ps1 = sse<v128<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  std::cout<< "using sse, 16bit" << std::endl;  
    //  base_t hadded = vectorlib::hadd<ps1, vector_base_t_granularity::value>::apply(sequence1); 
    //  std::cout << unsigned(hadded) << std::endl;
    // }


    // { //test min
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
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

    // { //test div
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(2,2); //2,4,6,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t divided = vectorlib::div<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(divided, i); //1,2,3,..
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test mod
    // using ps1 = avx2<v256<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(2,1); //2,3,4,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t modulo = vectorlib::mod<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(modulo, i); //0,1,0,...
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

    // { //test equal
    // using ps1 = avx2<v256<uint64_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //2,4,6,...,
    //  vector_t sequence2 = set_sequence<ps1, vector_base_t_granularity::value>(1,1);     //4,4,...,4
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  uint64_t equalnum = equal<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  std::cout << equalnum << std::endl;
    //  // uint64_t number1 = 0;
    //  // for(int i=0; i < 4; i++){
    //  //     number1 = extract_value<ps1,vector_base_t_granularity::value>(equalnum, i);
    //  //     std::cout << number1 << std::endl;
    //  // }
    // }

   // { //test manipulate
   //  using ps1 = avx2<v256<uint32_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
   //   std::cout<< "using avx2, 32bit" << std::endl;  
   //   vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); //2,3,4,5,6,7,8,1
   //   uint32_t number1 = 0;
   //   for(int i=0; i < 8; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
   //       std::cout << number1 << std::endl; 
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

    // { //test div
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(2,2); //2,4,6,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using avx2, 16bit" << std::endl;  
    //  vector_t divided = vectorlib::div<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(divided, i); //1,2,3,..
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

    // { //test shift_left_individual
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
    //  vector_t sequence2 = set_sequence<ps1, vector_base_t_granularity::value>(1,0); //1,1,1,...    
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t shifted = shift_left_individual<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
    //      std::cout << number1 << std::endl;
    //  }
    // }

    //  { //test manipulate
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,16
    //  std::cout<< "using avx2, 16bit" << std::endl;  
    //  vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); //2,3,4,5,6,7,8,9,...,16,1
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
    //      std::cout << number1 << std::endl; 
    //  }
    // }

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

    // { //test shift_left_individual
    // using ps1 = avx2<v256<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
    //  vector_t sequence2 = set_sequence<ps1, vector_base_t_granularity::value>(1,0); //1,1,1,...    
    //  std::cout<< "using avx2, 8bit" << std::endl;  
    //  vector_t shifted = shift_left_individual<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 32; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

    //   { //test shift_left
    // using ps1 = avx2<v256<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
    //  std::cout<< "using avx2, 8bit" << std::endl;  
    //  vector_t shifted = shift_left<ps1, vector_base_t_granularity::value>::apply(sequence1, 1); 
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 32; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

    //  { //test manipulate
    // using ps1 = avx2<v256<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,16
    //  std::cout<< "using avx2, 8bit" << std::endl;  
    //  vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); //2,3,4,5,6,7,8,9,...,16,1
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 32; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
    //      std::cout << number1 << std::endl; 
    //  }
    // }

    //tests for avx2<v256<uint32_t>>:

    // { //test add
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,3
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,1,1
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,3,4
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test sub
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,1,1
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t subtracted = sub<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(subtracted, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test min
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,3
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,2,2
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,2
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test mul
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,3
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,2,2
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t multiplied = mul<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,2,4,6
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(multiplied, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test div
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(3,3); //2,4,6,8
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t divided = vectorlib::div<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(divided, i); //1,2,3,..
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test mod
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(2,1); //2,3,4,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t modulo = vectorlib::mod<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(modulo, i); //0,1,0,...
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test shift_left_individual
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //1,1,1
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t shifted = shift_left_individual<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test equal, error
    // using ps1 = sse<v128<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(2,2); //2,4,6,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
    //  std::cout<< "using sse, 32bit" << std::endl;  
    //  vector_t equalnum = equal<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint32_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(equalnum, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

   // { //test manipulate
   //  using ps1 = sse<v128<uint32_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,
   //   std::cout<< "using sse, 32bit" << std::endl;  
   //   vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); 
   //   uint32_t number1 = 0;
   //   for(int i=0; i < 4; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
   //       std::cout << number1 << std::endl; 
   //   }
   //  }

    //tests for sse<v128<uint16_t>>:

    // { //test add
    // using ps1 = sse<v128<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using sse, 16bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,8
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

  // { //test min
  //   using ps1 = sse<v128<uint16_t>>;
  //   IMPORT_VECTOR_BOILER_PLATE(ps1)

  //    vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
  //    vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
  //    std::cout<< "using sse, 16bit" << std::endl;  
  //    vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,4,4,4,4
  //    uint16_t number1 = 0;
  //    for(int i=0; i < 8; i++){
  //        number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
  //        std::cout << number1 << std::endl;
  //    }
  //   }

    // { //test sub
    // using ps1 = sse<v128<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using sse, 16bit" << std::endl;  
    //  vector_t subtracted = sub<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,...,7
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(subtracted, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test mul
    // using ps1 = sse<v128<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using sse, 16bit" << std::endl;  
    //  vector_t multiplied = mul<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,2,4,6,8,10,12,14
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(multiplied, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

    // { //test shift_left_individual
    // using ps1 = sse<v128<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //1,1,1
    //  std::cout<< "using sse, 16bit" << std::endl;  
    //  vector_t shifted = shift_left_individual<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
    //      std::cout << number1 << std::endl;
    //  }
    // }

  // { //test manipulate
  //   using ps1 = sse<v128<uint16_t>>;
  //   IMPORT_VECTOR_BOILER_PLATE(ps1)

  //    vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,
  //    std::cout<< "using sse, 16bit" << std::endl;  
  //    vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); 
  //    uint32_t number1 = 0;
  //    for(int i=0; i < 8; i++){
  //        number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
  //        std::cout << number1 << std::endl; 
  //    }
  //   }
    //   { //test div
    // using ps1 = sse<v128<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,3); //2,4,6,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using sse, 16bit" << std::endl;  
    //  vector_t divided = vectorlib::div<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); 
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 8; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(divided, i); //1,2,3,..
    //      std::cout << number1 << std::endl;

    //  }
    // }

    //tests for sse<v128<uint8_t>>;

    // { //test add
    // using ps1 = sse<v128<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using sse, 8bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

   // { //test min
   //  using ps1 = sse<v128<uint8_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,
   //   vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(4);     //4,4,...,4
   //   std::cout<< "using sse, 8bit" << std::endl;  
   //   vector_t minimum = min<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,2,3,4,4,4,4,..
   //   uint8_t number1 = 0;
   //   for(int i=0; i < 16; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(minimum, i);
   //       std::cout << unsigned(number1) << std::endl;
   //   }
   //  }

    // { //test sub
    // using ps1 = sse<v128<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using sse, 8bit" << std::endl;  
    //  vector_t subtracted = sub<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,1,...,
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(subtracted, i);
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

    // { //test shift_left
    // using ps1 = sse<v128<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
    //  std::cout<< "using sse, 8bit" << std::endl;  
    //  vector_t shifted = shift_right<ps1, vector_base_t_granularity::value>::apply(sequence1, 1); 
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

    // { //test mul
    // using ps1 = sse<v128<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2);     //2,2,...,2
    //  std::cout<< "using sse, 8bit" << std::endl;  
    //  vector_t multiplied = mul<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,2,4,6,8,10,12,14
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 16; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(multiplied, i);
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }

  // { //test manipulate
  //   using ps1 = sse<v128<uint8_t>>;
  //   IMPORT_VECTOR_BOILER_PLATE(ps1)

  //    vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,
  //    std::cout<< "using sse, 8bit" << std::endl;  
  //    vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); 
  //    uint32_t number1 = 0;
  //    for(int i=0; i < 16; i++){
  //        number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
  //        std::cout << number1 << std::endl; 
  //    }
  //   }

  //tests for avx512<v512<...>>:
    // {
    // using ps= avx512<v512<uint64_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps)

    // vector_t sequence = set_sequence<ps, vector_base_t_granularity::value>(0,1);
    // uint64_t number = 0;
    // for(int i=0; i < 8; i++){
    //     number = extract_value<ps,vector_base_t_granularity::value>(sequence, i); //0,1,2,...,7
    //     std::cout << number << std::endl;
    // }
    // }

    // {
    // using ps= avx512<v512<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps)

    // vector_t sequence = set_sequence<ps, vector_base_t_granularity::value>(0,1);
    // uint8_t number = 0;
    // for(int i=0; i < 64; i++){
    //     number = extract_value<ps,vector_base_t_granularity::value>(sequence, i);
    //     std::cout << unsigned(number) << std::endl;
    // }
    // }
    // { //test hadd
    // using ps1 = avx2<v256<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  base_t hadded = hadd<ps1, vector_base_t_granularity::value>::apply(sequence1); 
    //  std::cout << unsigned(hadded) << std::endl;
    // }

    // { //test hadd
    // using ps1 = avx512<v512<uint32_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  base_t hadded = hadd<ps1, vector_base_t_granularity::value>::apply(sequence1); 
    //  std::cout << unsigned(hadded) << std::endl;
    // }

    // { //test hadd
    // using ps1= avx512<v512<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)
    // vector_t sequence1 = set1<ps1, vector_base_t_granularity::value>(1);
    //  //vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
    //  std::cout<< "using avx512, 16bit" << std::endl;  
    //  base_t hadded = vectorlib::hadd<ps1, vector_base_t_granularity::value>::apply(sequence1); 
    //  std::cout << unsigned(hadded) << std::endl;
    // }

    // { //test add
    // using ps1 = avx512<v512<uint16_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(0,1); //0,1,2,...,15
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 16bit" << std::endl;  
    //  vector_t added = add<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,16
    //  uint16_t number1 = 0;
    //  for(int i=0; i < 32; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(added, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }

   // { //test manipulate
   //  using ps1 = avx512<v512<uint16_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   //   vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,...,8
   //   vector_t manipulated = manipulate<ps1, vector_base_t_granularity::value>::rotate(sequence1); //2,3,4,5,6,7,8,1
   //   uint16_t number1 = 0;
   //   for(int i=0; i < 32; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(manipulated, i); 
   //       std::cout << number1 << std::endl; 
   //   }
   //  }

   //    { //test shift_left
   //  using ps1 = avx512<v512<uint16_t>>;
   //  IMPORT_VECTOR_BOILER_PLATE(ps1)

   // //  vector_t sequence1 = set_sequence<ps1, vector_base_t_granularity::value>(1,1); //1,2,3,4,...,
   //   vector_t sequence1 = set1<ps1, vector_base_t_granularity::value>(2); 
   //   std::cout<< "using avx2, 8bit" << std::endl;  
   //   vector_t shifted = shift_left<ps1, vector_base_t_granularity::value>::apply(sequence1, 1); 
   //   uint16_t number1 = 0;
   //   for(int i=0; i < 32; i++){
   //       number1 = extract_value<ps1,vector_base_t_granularity::value>(shifted, i); 
   //       std::cout << unsigned(number1) << std::endl;
   //   }
   //  }

    // { //test mul
    // using ps1 = avx512<v512<uint8_t>>;
    // IMPORT_VECTOR_BOILER_PLATE(ps1)

    //  vector_t sequence1 = set1<ps1, vector_base_t_granularity::value>(2); 
    //  vector_t sequence2 = set1<ps1, vector_base_t_granularity::value>(2); 
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  vector_t multiplied = mul<ps1, vector_base_t_granularity::value>::apply(sequence1, sequence2); //0,2,4,6,8,10,12,14
    //  uint8_t number1 = 0;
    //  for(int i=0; i < 65; i++){
    //      number1 = extract_value<ps1,vector_base_t_granularity::value>(multiplied, i);
    //      std::cout << unsigned(number1) << std::endl;
    //  }
    // }
}
 
