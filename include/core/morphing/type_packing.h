    #ifndef MORPHSTORE_CORE_MORPHING_TYPE_PACKING_H
#define MORPHSTORE_CORE_MORPHING_TYPE_PACKING_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>



#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>

#include <cstdint>
#include <cstring>

namespace morphstore {

    // ************************************************************************
    // Format
    // ************************************************************************
    
    template<class t_layout>//, class t_vector_extension>
    struct type_packing_f : public format {
       
        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues * sizeof(t_layout);
        }
        static const size_t m_BlockSize = 16;
       // IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)  

        //static const size_t m_BlockSize = vector_element_count::value;
        public:
        static const uint8_t * const m_maskCompr;
        static const uint8_t * const m_maskDecompr;
        static const uint8_t * const m_maskCompr_avx2;
        static const uint8_t * const m_maskDecompr_avx2;
        static const uint8_t * const m_maskCompr_avx512; 
        static const uint8_t * const m_permMaskCompr_avx512; 
        static const uint8_t * const m_maskDecompr_avx512;
        static const uint8_t * const m_permMaskDecompr_avx512;

        static const uint8_t * build_table_shuffle_mask_compr() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(16))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(16)

            );
#endif 
           // std::cout << "sizeof(t_layout) " <<sizeof(t_layout) << std::endl;
            if(sizeof(t_layout) == 1){ //hardcoded mask, for compression to 8 bit
            res[0] = 0;
            res[1] = 8;
            res[2] = 127;
            res[3] = 127;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 127;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            }

            if(sizeof(t_layout) == 2){ //hardcoded mask, for compression to 16 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 8;
            res[3] = 9;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 127;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            }

            if(sizeof(t_layout) == 4){ //hardcoded mask, for compression to 32 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 2;
            res[3] = 3;
            res[4] = 8;
            res[5] = 9;
            res[6] = 10;
            res[7] = 11;
            res[8] = 127;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            }


            return res;
        }

        static const uint8_t * build_table_shuffle_mask_compr_avx2() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                malloc(get_size_with_alignment_padding(32))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                malloc(32)

            );
#endif 
           // std::cout << "sizeof(t_layout) " <<sizeof(t_layout) << std::endl;
            if(sizeof(t_layout) == 1){ //hardcoded mask, for compression to 8 bit
            res[0] = 0;
            res[1] = 8;
            res[2] = 4;
            res[3] = 12;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 127;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            }

            if(sizeof(t_layout) == 2){ //hardcoded mask, for compression to 16 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 8;
            res[3] = 9;
            res[4] = 4;
            res[5] = 5;
            res[6] = 12;
            res[7] = 13;
            res[8] = 127;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            }

            if(sizeof(t_layout) == 4){ //hardcoded mask, for compression to 32 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 2;
            res[3] = 3;
            res[4] = 8;
            res[5] = 9;
            res[6] = 10;
            res[7] = 11;
            res[8] = 4;   
            res[9] = 5;
            res[10] = 6;
            res[11] = 7;
            res[12] = 12;
            res[13] = 13;
            res[14] = 14;
            res[15] = 15;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;          
            }


            return res;
        }


        static const uint8_t * build_table_shuffle_mask_compr_avx512() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                malloc(get_size_with_alignment_padding(64))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                malloc(64)

            );
#endif 
            //std::cout << "sizeof(t_layout) " <<sizeof(t_layout) << std::endl;
            if(sizeof(t_layout) == 1){ //hardcoded mask, shuffle for compression to 8 bit
            //std::cout<< "test1" << std::endl;
            res[0] = 0;
            res[1] = 2;
            res[2] = 4;
            res[3] = 6;
            res[4] = 8;
            res[5] = 10;
            res[6] = 12;
            res[7] = 14;             
            res[8] = 127;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 127;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 127;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 127;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 127;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;                        
         }

            if(sizeof(t_layout) == 2){ //hardcoded mask, for compression to 16 bit
            //std::cout<< "test2" << std::endl;
            res[0] = 0;
            res[1] = 127;
            res[2] = 4;
            res[3] = 127;
            res[4] = 8;
            res[5] = 127;
            res[6] = 12;
            res[7] = 127;             
            res[8] = 16;   
            res[9] = 127;
            res[10] = 20;
            res[11] = 127;
            res[12] = 24;
            res[13] = 127;
            res[14] = 28;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 127;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 127;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 127;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 127;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;  
         }

            if(sizeof(t_layout) == 4){ //hardcoded mask, for compression to 32 bit
            //std::cout<< "test3" << std::endl;
            res[0] = 0;
            res[1] = 127;
            res[2] = 1;
            res[3] = 127;
            res[4] = 4;
            res[5] = 127;
            res[6] = 5;
            res[7] = 127;             
            res[8] = 8;   
            res[9] = 127;
            res[10] = 9;
            res[11] = 127;
            res[12] = 12;
            res[13] = 127;
            res[14] = 13;
            res[15] = 127;
            res[16] = 16;
            res[17] = 127;
            res[18] = 17;
            res[19] = 127;
            res[20] = 20;
            res[21] = 127;
            res[22] = 21;
            res[23] = 127;
            res[24] = 24;
            res[25] = 127;   
            res[26] = 25;
            res[27] = 127;
            res[28] = 28;
            res[29] = 127;
            res[30] = 29;
            res[31] = 127;
            res[32] = 127;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 127;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 127;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 127;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;           
         }


            return res;
        }

        static const uint8_t * build_table_shuffle_permMaskCompr_avx512() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(64))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(64)

            );
#endif 
            //hardcoded mask, for compression to 8 bit
            res[0] = 0;
            res[1] = 127;
            res[2] = 4;
            res[3] = 127;
            res[4] = 8;
            res[5] = 127;
            res[6] = 12;
            res[7] = 127;             
            res[8] = 16;   
            res[9] = 127;
            res[10] = 20;
            res[11] = 127;
            res[12] = 24;
            res[13] = 127;
            res[14] = 28;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 127;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 127;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 127;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 127;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;   

            return res;
        }


        static const uint8_t * build_table_shuffle_mask_decompr() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(16))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(16)

            );
#endif 
           // std::cout << "sizeof(t_layout) " <<sizeof(t_layout) << std::endl;
            if(sizeof(t_layout) == 1){ //hardcoded mask, for decompression of 8 bit
            res[0] = 0;
            res[1] = 127;
            res[2] = 127;
            res[3] = 127;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 1;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            }

            if(sizeof(t_layout) == 2){ //hardcoded mask, for decompression of 16 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 127;
            res[3] = 127;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 2;   
            res[9] = 3;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            }

            if(sizeof(t_layout) == 4){ //hardcoded mask, for decompression of 32 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 2;
            res[3] = 3;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 4;   
            res[9] = 5;
            res[10] = 6;
            res[11] = 7;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            }


            return res;
        }

        static const uint8_t * build_table_shuffle_mask_decompr_avx2() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(32))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(32)

            );
#endif 
           // std::cout << "sizeof(t_layout) " <<sizeof(t_layout) << std::endl;
            if(sizeof(t_layout) == 1){ //hardcoded mask, for decompression of 8 bit
            res[0] = 0;
            res[1] = 127;
            res[2] = 127;
            res[3] = 127;
            res[4] = 1;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;
            res[8] = 2;   
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 3;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            }

            if(sizeof(t_layout) == 2){ //hardcoded mask, for decompression of 16 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 127;
            res[3] = 127;
            res[4] = 2;
            res[5] = 3;
            res[6] = 127;
            res[7] = 127;
            res[8] = 4;   
            res[9] = 5;
            res[10] = 127;
            res[11] = 127;
            res[12] = 6;
            res[13] = 7;
            res[14] = 127;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            }

            if(sizeof(t_layout) == 4){ //hardcoded mask, for decompression of 32 bit
            res[0] = 0;
            res[1] = 1;
            res[2] = 2;
            res[3] = 3;
            res[4] = 4;
            res[5] = 5;
            res[6] = 6;
            res[7] = 7;
            res[8] = 8;   
            res[9] = 9;
            res[10] = 10;
            res[11] = 11;
            res[12] = 12;
            res[13] = 13;
            res[14] = 14;
            res[15] = 15;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            }


            return res;
        }

        static const uint8_t * build_table_shuffle_mask_decompr_avx512() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(64))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(64)

            );
#endif 
            //std::cout << "sizeof(t_layout) " <<sizeof(t_layout) << std::endl;
            if(sizeof(t_layout) == 1){ //hardcoded mask, for shuffle for decompression to 8 bit
            //std::cout<< "test1" << std::endl;
            res[0] = 0;
            res[1] = 127;
            res[2] = 1;
            res[3] = 127;
            res[4] = 2;
            res[5] = 127;
            res[6] = 3;
            res[7] = 127;                
            res[8] = 4;   
            res[9] = 127;
            res[10] = 5;
            res[11] = 127;
            res[12] = 6;
            res[13] = 127;
            res[14] = 7;
            res[15] = 127;
            res[16] = 127;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 127;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 127;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 127;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 127;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 127;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;            
         }

            if(sizeof(t_layout) == 2){ //hardcoded mask, for decompression to 16 bit
            //std::cout<< "test2" << std::endl;
            res[0] = 0;
            res[1] = 127;
            res[2] = 127;
            res[3] = 127;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;           
            res[8] = 1;  
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 2;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 3;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 4;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 5;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 6;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 7;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;              
         }

            if(sizeof(t_layout) == 4){ //hardcoded mask, for decompression to 32 bit
            //std::cout<< "test3" << std::endl;
            res[0] = 0;
            res[1] = 127;
            res[2] = 1;
            res[3] = 127;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;           
            res[8] = 2;  
            res[9] = 127;
            res[10] = 3;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 4;
            res[17] = 127;
            res[18] = 5;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 6;
            res[25] = 127;   
            res[26] = 7;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 8;
            res[33] = 127;
            res[34] = 9;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 10;   
            res[41] = 127;
            res[42] = 11;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 12;
            res[49] = 127;
            res[50] = 13;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 14;
            res[57] = 127;   
            res[58] = 15;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;          
         }


            return res;
        }

        static const uint8_t * build_table_shuffle_permMask_decompr_avx512() {
#ifdef MSV_NO_SELFMANAGED_MEMORY
            uint8_t * res = create_aligned_ptr(reinterpret_cast<uint8_t *>(
                    malloc(get_size_with_alignment_padding(64))
            ));
#else
            uint8_t * res = reinterpret_cast<uint8_t *>(
                    malloc(64)

            );
#endif 
            //hardcoded mask, for decompression to 8 bit
            res[0] = 0;
            res[1] = 127;
            res[2] = 127;
            res[3] = 127;
            res[4] = 127;
            res[5] = 127;
            res[6] = 127;
            res[7] = 127;           
            res[8] = 1;  
            res[9] = 127;
            res[10] = 127;
            res[11] = 127;
            res[12] = 127;
            res[13] = 127;
            res[14] = 127;
            res[15] = 127;
            res[16] = 2;
            res[17] = 127;
            res[18] = 127;
            res[19] = 127;
            res[20] = 127;
            res[21] = 127;
            res[22] = 127;
            res[23] = 127;
            res[24] = 3;
            res[25] = 127;   
            res[26] = 127;
            res[27] = 127;
            res[28] = 127;
            res[29] = 127;
            res[30] = 127;
            res[31] = 127;
            res[32] = 4;
            res[33] = 127;
            res[34] = 127;
            res[35] = 127;
            res[36] = 127;
            res[37] = 127;
            res[38] = 127;
            res[39] = 127;
            res[40] = 5;   
            res[41] = 127;
            res[42] = 127;
            res[43] = 127;
            res[44] = 127;
            res[45] = 127;
            res[46] = 127;
            res[47] = 127;
            res[48] = 6;
            res[49] = 127;
            res[50] = 127;
            res[51] = 127;
            res[52] = 127;
            res[53] = 127;
            res[54] = 127;
            res[55] = 127;
            res[56] = 7;
            res[57] = 127;   
            res[58] = 127;
            res[59] = 127;
            res[60] = 127;
            res[61] = 127;
            res[62] = 127;
            res[63] = 127;             
            return res;
        }

    };
    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskCompr =
    build_table_shuffle_mask_compr();
  
    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskDecompr =
    build_table_shuffle_mask_decompr();

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskCompr_avx2 =
    build_table_shuffle_mask_compr_avx2();

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskDecompr_avx2 =
    build_table_shuffle_mask_decompr_avx2();    

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskCompr_avx512 =
    build_table_shuffle_mask_compr_avx512();    

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_permMaskCompr_avx512 =
    build_table_shuffle_permMaskCompr_avx512();  

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskDecompr_avx512 =
    build_table_shuffle_mask_decompr_avx512();  

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_permMaskDecompr_avx512 =
    build_table_shuffle_permMask_decompr_avx512();      
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    //without vectorisation:
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, type_packing_f<t_layout>, uncompr_f
    > {
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
        const size_t sizeByte = sizeof(t_layout);
        for(size_t i = 0; i < countLog; i++){
         memcpy(out8, in8, sizeByte); 
         in8 += 8;
         out8 +=sizeByte;
        }
       }
    };   


#ifdef SSE
    template<class t_layout>
    struct morph_batch_t<
            vectorlib::sse<vectorlib::v128<uint64_t> >, type_packing_f<t_layout>, uncompr_f
    > {

        using t_ve = vectorlib::sse<vectorlib::v128<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)  
        using dst_f = type_packing_f<t_layout>;             
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
        using namespace vectorlib;
        const base_t * inBase = reinterpret_cast<const base_t *>(in8);
        const base_t * mask_base = reinterpret_cast<const base_t *>(dst_f::m_maskCompr);  
        const size_t sizeByte = sizeof(t_layout);       
        if(sizeof(t_layout) == 8){ //if it "compresses" to 64 bit, give back the column unchanged 
            const size_t size = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, size);
            in8 += size;
            out8 += size;            
        }
        else{        
            vector_t const mask = load< //calls hardcoded mask
                            t_ve,
                            iov::ALIGNED,
                            vector_size_bit::value
                    >(mask_base);     
            for(size_t i = 0; i < countLog; i += vector_element_count::value){
                    store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                            reinterpret_cast<base_t *>(out8),
                            _mm_shuffle_epi8(
                                    load<
                                            t_ve,
                                            iov::ALIGNED,
                                            vector_size_bit::value
                                    >(inBase + i),
                                    mask                                                            
                            )
                    );
                    out8 += sizeByte*vector_element_count::value;
            }
            in8 += convert_size<uint64_t, uint8_t>(countLog);
        }
    }
};   
#endif       


#ifdef AVXTWO
    template<class t_layout>
    struct morph_batch_t<
            vectorlib::avx2<vectorlib::v256<uint64_t> >, type_packing_f<t_layout>, uncompr_f
    > {

        using t_ve = vectorlib::avx2<vectorlib::v256<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)  
        using dst_f = type_packing_f<t_layout>;             
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
        using namespace vectorlib;
        const base_t * inBase = reinterpret_cast<const base_t *>(in8);
        const base_t * mask_base = reinterpret_cast<const base_t *>(dst_f::m_maskCompr_avx2);  
        const size_t sizeByte = sizeof(t_layout);

        vector_t const mask = load< //calls hardcoded mask
                                t_ve,
                                iov::ALIGNED,
                                vector_size_bit::value
                        >(mask_base);
        if(sizeof(t_layout) == 8){ //if it "compresses" to 64 bit, give back the column unchanged 
            const size_t size = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, size);
            in8 += size;
            out8 += size;
        }
        else{
            for(size_t i = 0; i < countLog; i += vector_element_count::value){
                    store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                            reinterpret_cast<base_t *>(out8),
                            _mm256_shuffle_epi8(
                                _mm256_permutevar8x32_epi32(
                                        load<
                                                t_ve,
                                                iov::ALIGNED,
                                                vector_size_bit::value
                                        >(inBase + i),
                                        _mm256_set_epi32(0,0,0,0,6,2,4,0)                                                    
                                ),
                                mask
                        )
                    );
                    out8 += sizeByte*vector_element_count::value;
            }
            in8 += convert_size<uint64_t, uint8_t>(countLog);
        }
    }
};   
#endif    

#ifdef AVX512
    template<class t_layout>
    struct morph_batch_t<
            vectorlib::avx512<vectorlib::v512<uint64_t> >, type_packing_f<t_layout>, uncompr_f
    > {

        using t_ve = vectorlib::avx512<vectorlib::v512<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)  
     using dst_f = type_packing_f<t_layout>;             
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
        using namespace vectorlib;
        const base_t * inBase = reinterpret_cast<const base_t *>(in8);
        const base_t * mask_base = reinterpret_cast<const base_t *>(dst_f::m_maskCompr_avx512);  
        const base_t * permMask_base = reinterpret_cast<const base_t *>(dst_f::m_permMaskCompr_avx512);          
        const size_t sizeByte = sizeof(t_layout);                     
        if(sizeof(t_layout) == 8){ //if it "compresses" to 64 bit, give back the column unchanged 
            const size_t size = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, size);
            in8 += size;
            out8 += size;
        }
        else{
            vector_t const mask = load< //calls hardcoded mask
                                    t_ve,
                                    iov::ALIGNED,
                                    vector_size_bit::value
                            >(mask_base);
            vector_t const permMask = load< //calls hardcoded mask
                                    t_ve,
                                    iov::ALIGNED,
                                    vector_size_bit::value
                            >(permMask_base);               
            for(size_t i = 0; i < countLog; i += vector_element_count::value){
             if(sizeByte == 1){
                    store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                            reinterpret_cast<base_t *>(out8),
                            _mm512_shuffle_epi8(
                             _mm512_permutexvar_epi16(
                                        permMask,
                                         load<
                                                 t_ve,
                                                 iov::ALIGNED,
                                                 vector_size_bit::value
                                         >(inBase + i)
                             ), 
                             mask
                         )                                 
                    );               
             }                    
             else if(sizeByte == 2||4){
                    store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                            reinterpret_cast<base_t *>(out8),
                             _mm512_permutexvar_epi16(
                                         mask,
                                         load<
                                                 t_ve,
                                                 iov::ALIGNED,
                                                 vector_size_bit::value
                                         >(inBase + i)
                             )                                     
                    );
                }
                out8 += sizeByte*vector_element_count::value;
            }
            in8 += convert_size<uint64_t, uint8_t>(countLog);
           }
        }
    };   
#endif       
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    //without vectorisation:   
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, uncompr_f, type_packing_f<t_layout>
    > {
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
        const size_t sizeByte = sizeof(t_layout);            
        for(size_t i = 0; i < countLog; i++){
         memcpy(out8, in8, sizeByte); 
         in8 += sizeByte;
         out8 +=8;
        }        
       }
    };

#ifdef SSE
    template<class t_layout>
    struct morph_batch_t<
            vectorlib::sse<vectorlib::v128<uint64_t> >, uncompr_f, type_packing_f<t_layout>
    > {
        using t_ve = vectorlib::sse<vectorlib::v128<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = type_packing_f<t_layout>; 
        
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            const base_t * mask_base = reinterpret_cast<const base_t *>(src_f::m_maskDecompr); 
            const size_t sizeByte = sizeof(t_layout);            
        if(sizeof(t_layout) == 8){  
            const size_t size = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, size);
            in8 += size;
            out8 += size;
        }
        else{     
            vector_t const mask = load< //calls hardcoded mask
                                t_ve,
                                iov::ALIGNED,
                                vector_size_bit::value
                        >(mask_base);                                          
            for(size_t i = 0; i < countLog; i += vector_element_count::value) {
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase,
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(in8)),
                                mask
                        )
                );
                in8 += sizeByte*vector_element_count::value;
                outBase += vector_element_count::value;
            }
            
            out8 += convert_size<uint64_t, uint8_t>(countLog);
        }
    }
};
#endif

#ifdef AVXTWO
    template<class t_layout>
    struct morph_batch_t<
            vectorlib::avx2<vectorlib::v256<uint64_t> >, uncompr_f, type_packing_f<t_layout>
    > {
        using t_ve = vectorlib::avx2<vectorlib::v256<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = type_packing_f<t_layout>; 
        
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            const base_t * mask_base = reinterpret_cast<const base_t *>(src_f::m_maskDecompr_avx2); 
            const size_t sizeByte = sizeof(t_layout);  
        if(sizeof(t_layout) == 8){ 
            const size_t size = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, size);
            in8 += size;
            out8 += size;
        }
        else{                
            vector_t const mask = load< //calls hardcoded mask
                        t_ve,
                        iov::ALIGNED,
                        vector_size_bit::value
                >(mask_base);                                 
            for(size_t i = 0; i < countLog; i += vector_element_count::value) {
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase,
                        _mm256_permutevar8x32_epi32(
                            _mm256_shuffle_epi8(
                                    load<
                                            t_ve,
                                            iov::UNALIGNED,
                                            vector_size_bit::value
                                    >(reinterpret_cast<const base_t *>(in8)),
                                    mask
                            ),
                            _mm256_set_epi32(4,3,4,2,4,1,4,0) 
                        )
                );
                in8 += sizeByte*vector_element_count::value;
                outBase += vector_element_count::value;
            }
            
            out8 += convert_size<uint64_t, uint8_t>(countLog);
        }
    }
};
#endif

#ifdef AVX512
    template<class t_layout>
    struct morph_batch_t<
            vectorlib::avx512<vectorlib::v512<uint64_t> >, uncompr_f, type_packing_f<t_layout>
    > {
        using t_ve = vectorlib::avx512<vectorlib::v512<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = type_packing_f<t_layout>; 
        
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            const base_t * mask_base = reinterpret_cast<const base_t *>(src_f::m_maskDecompr_avx512); 
            const base_t * permMask_base = reinterpret_cast<const base_t *>(src_f::m_permMaskDecompr_avx512);            
            const size_t sizeByte = sizeof(t_layout);                                
        if(sizeof(t_layout) == 8){  
            const size_t size = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, size);
            in8 += size;
            out8 += size;
        }
        else{
            vector_t const mask = load< //calls hardcoded mask
                                t_ve,
                                iov::ALIGNED,
                                vector_size_bit::value
                        >(mask_base);   
            vector_t const permMask = load< //calls hardcoded mask
                                t_ve,
                                iov::ALIGNED,
                                vector_size_bit::value
                        >(permMask_base);              
            for(size_t i = 0; i < countLog; i += vector_element_count::value){
             if(sizeByte == 1){
                    store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                        outBase,
                        _mm512_permutexvar_epi16(
                                permMask,
                                _mm512_shuffle_epi8(
                                         load<
                                                 t_ve,
                                                 iov::ALIGNED,
                                                 vector_size_bit::value
                                         >(reinterpret_cast<const base_t *>(in8)),
                                         mask
                             )
                         )                                 
                    );               
             }
             else if(sizeByte == 2||4){
                    store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                            outBase,
                             _mm512_permutexvar_epi16(
                                        mask,
                                         load<
                                                 t_ve,
                                                 iov::ALIGNED,
                                                 vector_size_bit::value
                                         >(reinterpret_cast<const base_t *>(in8))
                             )                                     
                    );
                }
                in8 += sizeByte*vector_element_count::value;
                outBase += vector_element_count::value;            
            }
            out8 += convert_size<uint64_t, uint8_t>(countLog);
           }
        }
    };   
#endif  
   
    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
#ifdef SSE
    template<
            class t_layout,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            vectorlib::sse<vectorlib::v128<uint64_t> >,
            type_packing_f<t_layout>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = vectorlib::sse<vectorlib::v128<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = type_packing_f<t_layout>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<
                        vectorlib::sse<vectorlib::v128<uint64_t> >,
                        t_extra_args ...
                >::state_t & p_State
        ) {
            if(sizeof(t_layout) == 8){
                const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);

                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value)
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            vectorlib::load<
                                    t_ve,
                                    vectorlib::iov::ALIGNED,
                                    vector_base_t_granularity::value
                            >(inBase + i),
                            p_State
                    );
                
                p_In8 = reinterpret_cast<const uint8_t *>(inBase + p_CountInLog);               
            }
            else{
                using namespace vectorlib;
                const base_t * mask_base = reinterpret_cast<const base_t *>(in_f::m_maskDecompr); 
                const size_t sizeByte = sizeof(t_layout);             
                vector_t const mask = load< //calls hardcoded mask
                                    t_ve,
                                    iov::ALIGNED,
                                    vector_size_bit::value
                            >(mask_base);
                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value) {
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            _mm_shuffle_epi8(
                                    load<
                                            t_ve,
                                            iov::UNALIGNED,
                                            vector_size_bit::value
                                    >(reinterpret_cast<const base_t *>(p_In8)),
                                    mask
                            ),
                            p_State
                    );
                    p_In8 += sizeByte*vector_element_count::value;
                }
            }
        }
    };
#endif


#ifdef AVXTWO
    template<
            class t_layout,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            vectorlib::avx2<vectorlib::v256<uint64_t> >,
            type_packing_f<t_layout>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = vectorlib::avx2<vectorlib::v256<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = type_packing_f<t_layout>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<
                        vectorlib::avx2<vectorlib::v256<uint64_t> >,
                        t_extra_args ...
                >::state_t & p_State
        ) {
            if(sizeof(t_layout) == 8){
                const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);

                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value)
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            vectorlib::load<
                                    t_ve,
                                    vectorlib::iov::ALIGNED,
                                    vector_base_t_granularity::value
                            >(inBase + i),
                            p_State
                    );
                
                p_In8 = reinterpret_cast<const uint8_t *>(inBase + p_CountInLog);               
            }
            else{
                using namespace vectorlib;
                const base_t * mask_base = reinterpret_cast<const base_t *>(in_f::m_maskDecompr_avx2); 
                const size_t sizeByte = sizeof(t_layout);             
                vector_t const mask = load< //calls hardcoded mask
                                    t_ve,
                                    iov::ALIGNED,
                                    vector_size_bit::value
                            >(mask_base);
                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value) {
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            _mm256_permutevar8x32_epi32(
                                _mm256_shuffle_epi8(
                                        load<
                                                t_ve,
                                                iov::UNALIGNED,
                                                vector_size_bit::value
                                        >(reinterpret_cast<const base_t *>(p_In8)),
                                        mask
                                ),
                                _mm256_set_epi32(4,3,4,2,4,1,4,0)                       
                            ),
                            p_State
                    );
                    p_In8 += sizeByte*vector_element_count::value;
                }
            }
        }
    };
#endif


#ifdef AVX512
    template<
            class t_layout,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            vectorlib::avx512<vectorlib::v512<uint64_t> >,
            type_packing_f<t_layout>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = vectorlib::avx512<vectorlib::v512<uint64_t> >;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = type_packing_f<t_layout>;
        
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<
                        vectorlib::avx512<vectorlib::v512<uint64_t> >,
                        t_extra_args ...
                >::state_t & p_State
        ) {
            if(sizeof(t_layout) == 8){
                const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);

                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value)
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            vectorlib::load<
                                    t_ve,
                                    vectorlib::iov::ALIGNED,
                                    vector_base_t_granularity::value
                            >(inBase + i),
                            p_State
                    );
                
                p_In8 = reinterpret_cast<const uint8_t *>(inBase + p_CountInLog);               
            }
            else{
                using namespace vectorlib;
                const base_t * mask_base = reinterpret_cast<const base_t *>(in_f::m_maskDecompr_avx512); 
                const base_t * permMask_base = reinterpret_cast<const base_t *>(in_f::m_permMaskDecompr_avx512);            
                const size_t sizeByte = sizeof(t_layout);             
                vector_t const mask = load< //calls hardcoded mask
                                    t_ve,
                                    iov::ALIGNED,
                                    vector_size_bit::value
                            >(mask_base);
                vector_t const permMask = load< //calls hardcoded mask
                                t_ve,
                                iov::ALIGNED,
                                vector_size_bit::value
                        >(permMask_base);                              
                for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value) {
                    if(sizeByte == 1){    
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            _mm512_permutexvar_epi16(
                                permMask,
                                _mm512_shuffle_epi8(
                                         load<
                                                 t_ve,
                                                 iov::ALIGNED,
                                                 vector_size_bit::value
                                         >(reinterpret_cast<const base_t *>(p_In8)),
                                         mask
                                )
                            ), 
                            p_State
                    );
                }
                else if(sizeByte == 2||4){
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                             _mm512_permutexvar_epi16(
                                        mask,
                                         load<
                                                 t_ve,
                                                 iov::ALIGNED,
                                                 vector_size_bit::value
                                         >(reinterpret_cast<const base_t *>(p_In8))
                             ),
                             p_State                                     
                    );
                }
                p_In8 += sizeByte*vector_element_count::value;
                }
            }
        }
    };
#endif

}
#endif

