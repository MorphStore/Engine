#ifndef MORPHSTORE_CORE_MORPHING_TYPE_PACKING_VERTICAL_H
#define MORPHSTORE_CORE_MORPHING_TYPE_PACKING_VERTICAL_H

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
    	struct type_packing_vertical_f : public format {

        static size_t get_size_max_byte(size_t p_CountValues) {
            return p_CountValues * sizeof(t_layout);
        }
        static const size_t m_BlockSize = 8;
        template<class t_ve>
        static void print_vector(t_ve& tmp, const char* tag){
         	std::cout << tag<< ":";
      
    	  	uint8_t number1 = _mm_extract_epi8(tmp,0);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,1);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,2);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,3);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,4);
         	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,5);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,6);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,7);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,8);
         	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,9);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,10);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,11);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,12);
         	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,13);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,14);
          	std::cout << std::hex << unsigned(number1) << " ";
			number1 = _mm_extract_epi8(tmp,15);
          	std::cout << std::hex << unsigned(number1) << std:: endl;

   //      	uint8_t number1 = _mm256_extract_epi8(tmp,0);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,1);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,2);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,3);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,4);
   //       	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,5);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,6);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,7);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,8);
   //       	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,9);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,10);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,11);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,12);
   //       	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,13);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,14);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,15);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,16);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,17);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,18);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,19);
   //       	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,20);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,21);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,22);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,23);
   //       	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,24);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,25);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,26);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,27);
   //       	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,28);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,29);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,30);
   //        	std::cout << std::hex << unsigned(number1) << " ";
			// number1 = _mm256_extract_epi8(tmp,31);
   //        	std::cout << std::hex << unsigned(number1) << std:: endl;

          	}

	};   

    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, type_packing_vertical_f<t_layout>, uncompr_f
    > {

        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
		using dst_l = type_packing_vertical_f<t_layout>;


        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countInLog
        ) {
            using namespace vectorlib;
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);
            
       		//const size_t sizeByte = sizeof(t_layout);
            const size_t blockSizeVec = sizeof(base_t) / sizeof(t_layout);//vector_size_byte::value / (sizeByte * 2) ;
            const size_t byte_to_bit = 8;
            //std::cout << "sizeByte" << sizeByte << std::endl;
            // std::cout << "blockSizeVec" << blockSizeVec << std::endl;
            // std::cout << "vector_size_byte::value" << vector_size_byte::value << std::endl;

            for(size_t i = 0; i < countInLog;) {
                vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);

          		//dst_l::print_vector(tmp, "load");

                inBase += vector_element_count::value;
                i += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    vector_t next_loaded = load<
		                                            t_ve,
		                                            iov::ALIGNED,
		                                            vector_size_bit::value
		                                    >(inBase);
		            vector_t shifted = shift_left<t_ve>::apply(
				                                    next_loaded,
				                                    k * byte_to_bit );
           			//dst_l::print_vector(next_loaded, "next_loaded");
           			//dst_l::print_vector(shifted, "shifted");
                    tmp = bitwise_or<t_ve>(
                            tmp,
                            shifted
                    );
           			//dst_l::print_vector(tmp, "result");
           			inBase += vector_element_count::value;
           			i += vector_element_count::value;
                }

                //dst_l::print_vector(tmp, "store");
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, tmp
                );
                outBase += vector_element_count::value;
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
            out8 = reinterpret_cast<uint8_t *>(outBase);
        }
    };


    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    template<class t_vector_extension, class t_layout>
    struct morph_batch_t<
            t_vector_extension, uncompr_f, type_packing_vertical_f<t_layout>
    > {

        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
		using src_l = type_packing_vertical_f<t_layout>;
        static const vector_t mask;
        static void apply(
                const uint8_t * & in8, uint8_t * & out8, size_t countLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);
            base_t * outBase = reinterpret_cast<base_t *>(out8);            
       		//const size_t sizeByte = sizeof(t_layout);
            const size_t blockSizeVec = sizeof(base_t) / sizeof(t_layout);//vector_size_byte::value / (sizeByte * 2) ;
            const size_t byte_to_bit = 8;

            for(size_t i = 0; i < countLog;) {
                const vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        outBase, bitwise_and<t_ve>(mask, tmp)
                );
                outBase += vector_element_count::value;
                i += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            outBase,
                            bitwise_and<t_ve>(
                                    mask,
                                    shift_right<t_ve>::apply(tmp, k * byte_to_bit )
                            )
                    );
                    outBase += vector_element_count::value;
           			i += vector_element_count::value;                    
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
            out8 = reinterpret_cast<uint8_t *>(outBase);
        }
    };

    template<class t_vector_extension, class t_layout>
    const typename t_vector_extension::vector_t morph_batch_t<
            t_vector_extension,
            uncompr_f,
            type_packing_vertical_f<t_layout>
    >::mask = vectorlib::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
            //bitwidth_max<typename t_vector_extension::base_t>(std::numeric_limits<uint64_t>::digits)
            bitwidth_max<typename t_vector_extension::base_t>(8) 
    );

    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
    template<
            class t_layout,
            class t_vector_extension,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >
    class decompress_and_process_batch<
            t_vector_extension,
            type_packing_vertical_f<t_layout>,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        using src_l = type_packing_f<t_layout>;       
        static const vector_t mask;
        public:
        static void apply(
                const uint8_t * & in8,
                size_t countInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
            using namespace vectorlib;        
            const base_t * inBase = reinterpret_cast<const base_t *>(in8);  
       		//const size_t sizeByte = sizeof(t_layout);
            const size_t byte_to_bit = 8;        
            const size_t blockSizeVec = sizeof(base_t) / sizeof(t_layout);//vector_size_byte::value / (sizeByte * 2) ;
            
            for(size_t i = 0; i < countInLog;) { 
                const vector_t tmp = load<
                        t_ve, iov::ALIGNED, vector_size_bit::value
                >(inBase);
                inBase += vector_element_count::value;
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        bitwise_and<t_ve>(mask, tmp), opState
                );
                i += vector_element_count::value;
                for(size_t k = 1; k < blockSizeVec; k++) {
                    t_op_vector<t_ve, t_extra_args ...>::apply(
                            bitwise_and<t_ve>(
                                    mask,
                                    shift_right<t_ve>::apply(tmp, k * byte_to_bit)
                            ),
                            opState
                    );
                    i += vector_element_count::value;
                }
            }
            
            in8 = reinterpret_cast<const uint8_t *>(inBase);
        }
    };
    
    template<
            class t_layout,
            class t_vector_extension,
            template<class, class...> class t_op_vector,
            class ... t_extra_args
    >    const typename t_vector_extension::vector_t decompress_and_process_batch<
            t_vector_extension,
            type_packing_vertical_f<t_layout>,
            t_op_vector,
            t_extra_args ...
    >::mask = vectorlib::set1<
            t_vector_extension,
            t_vector_extension::vector_helper_t::granularity::value
    >(
    	   // bitwidth_max<typename t_vector_extension::base_t>(std::numeric_limits<uint64_t>::digits)
            bitwidth_max<typename t_vector_extension::base_t>(8)
    );
     


}
#endif