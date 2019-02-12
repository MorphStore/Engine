/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/persistence/binary_io.h"
#include "../../include/core/storage/column.h"
#include "../../include/core/utils/equality_check.h"
#include "../../include/core/morphing/static_vbp.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <bitset>
#include <immintrin.h>
#include <emmintrin.h>
#include <limits.h>


using namespace morphstore;

template <unsigned bw, class T>
auto selectCompressed(T * col, T predicate, size_t values){
    int count_results = 0;

    
    //create result column
    auto resultCol = new column< uncompr_f >(values);
    uint64_t * resultdata = resultCol->data( );

    //compress predicate
    std::bitset<128> * constBS= new std::bitset<128>();
    constBS[0]=(predicate);
    for (int i=bw;i<128-(128%bw);i++){
        constBS[0][i]=constBS[0][i%bw];
    }
        
    __m128i constraint_128=(__m128i)_mm_loadu_pd(&(((const double *)constBS)[0]));
    __m128i zwres;
    __mmask8 comp_mask=_mm_cmpeq_epi64_mask(_mm_set1_epi32(1),_mm_set1_epi32(1));
    
    //Create masks for comparison
    std::bitset<128> * constBS2 = new std::bitset<128> [128/bw];
    memset((void *)constBS2,0,128/bw*128);//set everything to 0
    for (int k=0;k<(128/bw)/2;k++){
            for (int j=0;j<bw;j++) constBS2[k].set((k*bw)+j);
    }
    for (int k=(128/bw)/2;k<(128/bw);k++){
        for (int j=0;j<bw;j++) constBS2[k].set(64+((k-128/bw)*bw)+j);
    }
    __m128i * vgl=(__m128i *) constBS2;
    
    //number of required 128-bit loads
    int input_cnt=values/(int)(128/bw);
    
    //value sper 64-bit
    int half=(int)(64/bw);
 
    //start comparison
    for (int i=0;i<input_cnt;i++) {
        zwres=_mm_xor_si128((__m128i)_mm_loadu_si128(&(((__m128i *)col)[i])),constraint_128);
        
        //Do the evaluation for every codeword in vector register
        for (int nrcodes=0;nrcodes<64/bw*2;nrcodes++){

            if (comp_mask ==_mm_cmpeq_epi64_mask(vgl[nrcodes],_mm_andnot_si128(zwres,_mm_loadu_si128(&(vgl[nrcodes]))))) {
                              
                if ( nrcodes < half ){
                    resultdata[count_results]=(i*(int)(128/bw))+(nrcodes*2)+1;
                }else{
                    resultdata[count_results]=(i*(int)(128/bw))+(2*(nrcodes-half)+1);
                }
              
                count_results++;
            }
            
        }
             
    }
    //end comparison
    
    
    resultCol->count_values(count_results);
    resultCol->size_used_byte(count_results * sizeof(uint64_t));
 
    return resultCol;
}


//TODO column statt ptr und values �bergeben?
template <class T>
auto Select(T * col, T predicate, size_t values){
    
    auto resultCol = new column< uncompr_f >( values );
    uint64_t * resultdata = resultCol->data( );
    
    size_t count_results=0;
    
    for (size_t i=0; i<values; i++){
        if (col[i] == predicate) {
            resultdata[count_results]=i;
            count_results++;
        }
    }
    
    resultCol->count_values(count_results);
    resultCol->size_used_byte(count_results * sizeof(uint64_t));

    return resultCol;
}

int main( void ) {

    cout << "Test filter started" << endl;
      
    cout << "Test filter uncompressed started" << endl;
    
    // Parameters.
    const size_t origCountValues = 128 * 1000;
    const size_t origSizeUsedByte = origCountValues * sizeof( uint64_t );
    uint64_t predicate=7;
        
    // Create the column.
    auto origColUncompressed = new column< uncompr_f >( origSizeUsedByte );
    uint64_t * origData = origColUncompressed->data( );
    for( unsigned i = 0; i < origCountValues; i++ )
        origData[ i ] = i%32;
    
    //K�nnen wir das in set_*/get_* umbenennen?
    origColUncompressed->count_values( origCountValues );
    origColUncompressed->size_used_byte( origSizeUsedByte );
    
    //Do Select
    column< uncompr_f > * result = Select<uint64_t>(origData,predicate,origCountValues);
    
    std::cout << "Found " << result->count_values() << " values" << std::endl;
    for (size_t i=0; i<result->count_values() && i< 5;i++) std::cout << "result: " << result->data()[i] << endl;
    std::cout << "Test filter uncompressed finished" << std::endl;
    
    
    
    cout << "Test filter compressed started" << endl;
       
    auto comprCol = new column< static_vbp_f< 8 > >( origSizeUsedByte );
    
    //Compress data
    morph( origColUncompressed, comprCol );

    //do Select
    result = selectCompressed< 8, uint64_t >(comprCol->data(),predicate,origCountValues);
    
    std::cout << "Found " << result->count_values() << " values" << std::endl;
    for (size_t i=0; i<result->count_values() && i< 5;i++) std::cout << "result: " << result->data()[i] << endl;
    std::cout << "Test filter compressed finished" << std::endl;
   return 0;
}