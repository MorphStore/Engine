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
    
    template<class t_layout>
    struct type_packing_f : public format {
       
        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues * sizeof(t_layout);
        }
        static const size_t m_BlockSize = 1;
   		public:
   		static const uint8_t * const m_maskCompr;
   		static const uint8_t * const m_maskDecompr;




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
            res[16] = 127;
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
            res[16] = 127;
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
            res[16] = 127;
    	    }


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
            res[16] = 127;
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
            res[16] = 127;
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
            res[16] = 127;
    	    }


            return res;
        }


    };

    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskCompr =
    build_table_shuffle_mask_compr();
  
    template<class t_layout>
    const uint8_t * const type_packing_f<t_layout>::m_maskDecompr =
    build_table_shuffle_mask_decompr();

    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    //without vectorisation:
    // template<class t_vector_extension, class t_layout>
    // struct morph_batch_t<
    //         t_vector_extension, type_packing_f<t_layout>, uncompr_f
    // > {
    //     static void apply(
    //             const uint8_t * & in8, uint8_t * & out8, size_t countLog
    //     ) {
    //     const size_t sizeByte = sizeof(t_layout);
    //     for(size_t i = 0; i < countLog; i++){
    //     	memcpy(out8, in8, sizeByte); 
    //     	in8 += 8;
    //     	out8 +=sizeByte;
    //     }
    //    }
    // };   


#ifdef SSE
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, type_packing_f<t_layout>, uncompr_f
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

        for(size_t i = 0; i < countLog; i += vector_element_count::value){
                store<t_ve, iov::UNALIGNED, vector_size_bit::value>(
                        reinterpret_cast<base_t *>(out8),
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(inBase + i),
                                load< //calls hardcoded mask
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(mask_base)                                                            
                        )
                );
                out8 += sizeByte*2;

        }
            in8 += convert_size<uint64_t, uint8_t>(countLog);

       }
    };   
#endif    

    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    //without vectorisation:   
    // template<class t_vector_extension, class t_layout>
    // struct morph_batch_t<
    //         t_vector_extension, uncompr_f, type_packing_f<t_layout>
    // > {
    //     static void apply(
    //             const uint8_t * & in8, uint8_t * & out8, size_t countLog
    //     ) {
    //     const size_t sizeByte = sizeof(t_layout);        	
    //     for(size_t i = 0; i < countLog; i++){
    //     	memcpy(out8, in8, sizeByte); 
    //     	in8 += sizeByte;
    //     	out8 +=8;
    //     }        
    //    }
    // };

#ifdef SSE
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, uncompr_f, type_packing_f<t_layout>
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
                       
            for(size_t i = 0; i < countLog; i += vector_element_count::value) {
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase,
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(in8)),
                                load< //calls hardcoded mask
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(mask_base)
                        )
                );
                in8 += sizeByte*2;
                outBase += vector_element_count::value;
            }
            
            out8 += convert_size<uint64_t, uint8_t>(countLog);
        }
    };
#endif


    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
#ifdef SSE
    template<
            class t_vector_extension,
            class t_layout,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >
    struct decompress_and_process_batch<
            t_vector_extension,
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
                        t_vector_extension,
                        t_extra_args ...
                >::state_t & p_State
        ) {

            using namespace vectorlib;
            const base_t * mask_base = reinterpret_cast<const base_t *>(in_f::m_maskDecompr); 
            const size_t sizeByte = sizeof(t_layout);             

            for(size_t i = 0; i < p_CountInLog; i += vector_element_count::value) {
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        _mm_shuffle_epi8(
                                load<
                                        t_ve,
                                        iov::UNALIGNED,
                                        vector_size_bit::value
                                >(reinterpret_cast<const base_t *>(p_In8)),
                                load< //calls hardcoded mask
                                        t_ve,
                                        iov::ALIGNED,
                                        vector_size_bit::value
                                >(mask_base)
                        ),
                        p_State
                );
                p_In8 += sizeByte*2;
            }
        }
    };
#endif


}