/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   intersect_compr.h
 * Author: Annett
 *
 * Created on 20. Februar 2020, 11:35
 */

#ifndef INTERSECT_COMPR_H
#define INTERSECT_COMPR_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/write_iterator.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <tuple>
#include <type_traits>

#include <cstdint>
#include <cstring>

namespace morphstore {
using namespace vectorlib;

template<class VectorExtension, class t_out_data_f, class t_in_data_f>
   
   struct intersect_processing_unit_wit {
     IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         base_t * data2ptr;
         size_t data2leftValues;
         unsigned offset;
         selective_write_iterator<VectorExtension, t_out_data_f> m_Wit;
         typename random_read_access<VectorExtension, t_in_data_f>::type m_Rra;
         state_t(uint8_t * p_OutData, const base_t * p_InData) :
                m_Wit(p_OutData), m_Rra(p_InData)
        {
             offset = 0;
            //
        }
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      apply(
         vector_t const p_Data1Vector,
         state_t    &    p_State
      ) {
                  
         vector_mask_t m_MaskLess;
         vector_mask_t m_MaskEqual;
         
         size_t rightLeftValues=p_State.data2leftValues;
        
         vector_t Data1Vector;
                  
         vector_t sequence;
         vector_t data2Vector;
         base_t ad;
         unsigned match_helper=0;
         bool match=false;
         unsigned offset=p_State.offset;
         base_t cur;
         unsigned step;
         if (offset>p_State.data2leftValues) {return;}
         
         for (unsigned i=0;i<vector_element_count::value;i++){
            //std::cout << "i: " << i << "\n";
            //offset=0;
            match=false;
            //current_endR=(base_t*)endInPosR;
            cur=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(p_Data1Vector,i);
            Data1Vector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(cur);
            rightLeftValues=p_State.data2leftValues-offset;
            
            
            while(!match && (rightLeftValues > (int) vector_element_count::value) ){
                
                    step=(unsigned)((rightLeftValues)/(vector_element_count::value+1));
                    sequence = vectorlib::set_sequence<VectorExtension,vector_base_t_granularity::value>(offset+step,step);
                    //data2Vector = vectorlib::gather<VectorExtension, vector_base_t_granularity::value,vector_base_t_granularity::value/8>(p_Data2Ptr,sequence);
                    data2Vector = p_State.m_Rra.get(sequence);
                    m_MaskEqual      = vectorlib::equal_t<VectorExtension>::apply(data2Vector, Data1Vector);
                    m_MaskLess   = vectorlib::less_t<VectorExtension>::apply(data2Vector, Data1Vector);

                 if (m_MaskEqual==0) {//No matches

                    if (m_MaskLess == 0){
                        //std::cout << "\t enter state eq=0 lt=0\n";
                   
                       rightLeftValues=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence,0)-offset;
                       
                    }

                    else {

                        match_helper=vectorlib::count_matches_t<VectorExtension>::apply(m_MaskLess);

                        ad=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence, match_helper-1);
                        
                        
                     
                        //TODO left values aus distance extracts berechnen
                        
                        if (match_helper==vector_element_count::value) {
                            rightLeftValues=p_State.data2leftValues-ad;
                           
                        }else{
                            rightLeftValues=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence,0)-offset;
                        }
                        
                        offset=(ad+1);

                    }

                  
                 }//repeat from step 2
                 else{//there are matches
                  //  std::cout << "\t enter state eq!=0\n"; 
                    match=true;
                    
                    //p_State.m_Wit.write(p_Data1Vector, m_MaskEqual);
                    p_State.m_Wit.write(Data1Vector, 1);

                    match_helper=vectorlib::count_matches_t<VectorExtension>::apply(m_MaskLess);
                    if (match_helper > 0){
                        ad=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence,match_helper-1);
                        offset=(ad+1);
                    }else{
                        ad=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence,0);
                        offset=(ad+1);
                    }
                    
                    
                   // std::cout << "\t\t  ad: " << ad << ", new offset: " << offset << "\n";
                }
                    
            }
              
            
            
             //Go through the rest of every iteration sequentially if no match was found, yet
             if (((rightLeftValues) <= (int) vector_element_count::value) && (match==false)) {
                 // std::cout << "\t search in scalar rest\n";
                 sequence = vectorlib::set_sequence<VectorExtension,vector_base_t_granularity::value>(offset,1);
                 data2Vector = p_State.m_Rra.get(sequence); 
                 m_MaskEqual      = vectorlib::equal_t<VectorExtension>::apply(data2Vector, Data1Vector);
                 m_MaskLess   = vectorlib::less_t<VectorExtension>::apply(data2Vector, Data1Vector);
                 
                 if (m_MaskEqual!=0) {
                     //p_State.m_Wit.write(p_Data1Vector, m_MaskEqual);
                     p_State.m_Wit.write(Data1Vector, 1);
                     
                     match_helper=vectorlib::count_matches<VectorExtension>::apply(m_MaskLess);
                    if (match_helper > 0){
                        ad=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence,match_helper-1);
                        offset=(ad+1);
                    }else{
                        ad=vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(sequence,0);
                        offset=(ad+1);
                    }
                     
                     match=true;
                  //   std::cout << "\t\t enter state eq!=0, last register\n";
                 }
               //  rightLeftValues=0;
                 
                 
             }
   
          //  if (match == false) offset=0; 
         }
        
        p_State.offset = offset;
        return;
      }
};

template<class VectorExtension, class t_out_data_f, class t_in_pos_l_f, class t_in_pos_r_f>
   class my_intersect_wit_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
   public:
      MSV_CXX_ATTRIBUTE_FORCE_INLINE 
      static
      const column<t_out_data_f> *
      apply(
         column< t_in_pos_l_f > const * const p_Data1Column,
         column< t_in_pos_r_f > const * const p_Data2Column,
         const size_t p_OutPosCountEstimate = 0
      ) {
          
          if(p_Data2Column->template prepare_for_random_access<VectorExtension>()){;}
            // @todo It would be nice to use warn() from the logger, but this
            // always outputs to cout, which interferes with our checks of the
            // program's output.
           /* std::cerr
                    << "[warn]: second intersect-operator's input data column was "
                       "prepared for random access within the "
                       "intersect-operator, which might corrupt measurements of "
                       "the operator's execution time"
                    << std::endl;*/
         const size_t inData1Count = p_Data1Column->get_count_values();
         const size_t inData2Count = p_Data2Column->get_count_values();
         
         //uint8_t  const * inData1Ptr = p_Data1Column->get_data( );
         base_t * inData2Ptr = p_Data2Column->get_data( );

         uint8_t const *         inDataPtr            = p_Data1Column->get_data();
         uint8_t const * const   startDataPtr         = inDataPtr;
         //size_t  const           inDataCountLog       = p_Data1Column->get_count_values();
         size_t  const           inDataCountLog       = p_Data1Column->get_count_values_compr();
         size_t  const           inDataSizeComprByte  = p_Data1Column->get_size_compr_byte();
         size_t  const           inDataSizeUsedByte   = p_Data1Column->get_size_used_byte();
         
         
         size_t const sizeByte =
            bool(p_OutPosCountEstimate)
            ? (p_OutPosCountEstimate * sizeof(base_t))
            : std::min(
                    get_size_max_byte_any_len<t_out_data_f>(inData1Count),
                    get_size_max_byte_any_len<t_out_data_f>(inData2Count)
            );
         auto outDataCol = new column<t_out_data_f>(sizeByte);
         uint8_t * outDataPtr = outDataCol->get_data( );
         const uint8_t * const initOutData = outDataPtr;
         size_t outSizeComprByte;
         
        typename intersect_processing_unit_wit<
                VectorExtension, t_out_data_f, t_in_pos_r_f
        >::state_t witState(outDataPtr,inData2Ptr);
        witState.data2ptr=inData2Ptr;
        witState.data2leftValues=p_Data2Column->get_count_values();


            //size_t const inSizeRestByte = startDataPtr + inDataSizeUsedByte - inDataPtr;
            /*size_t const inDataSizeUncomprVecByte = round_down_to_multiple(
               inSizeRestByte, vector_size_byte::value
            );*/
         
         inDataPtr = p_Data1Column->get_data();
         
         decompress_and_process_batch<
            VectorExtension,
            t_in_pos_l_f,
            intersect_processing_unit_wit,
            t_out_data_f,
            t_in_pos_r_f
         >::apply(
            inDataPtr,inDataCountLog, witState
         );
         
         //Get uncompressed size:
         inDataPtr = p_Data1Column->get_data_uncompr_start();
         size_t outCountLog;
         //Finish if there is no uncompressed rest
         if(inDataSizeComprByte == inDataSizeUsedByte || witState.offset >=  p_Data2Column->get_count_values()) {
             //std::cout << "no uncomressed rest\n";
           std::tie(
                    outSizeComprByte, std::ignore, outDataPtr
            ) = witState.m_Wit.done();
         outCountLog = witState.m_Wit.get_count_values();
         
         //std::cout << "\nfinish compressed vectorized batch, inCount: " << inDataCountLog << ", outCount:" << outCountLog << "\n";
         }else{
           //  std::cout << "\nfinish compressed vectorized batch, inCount: " << inDataCountLog << ", outCount:" << witState.m_Wit.get_count_values() << "\n";
           //  std::cout << "uncomressed rest\n";
             //do uncompressed rest
             inDataPtr = p_Data1Column->get_data_uncompr_start();
             const size_t inSizeRestByte = startDataPtr + inDataSizeUsedByte - inDataPtr;
             //const size_t inSizeRestByte = startDataPtr + inDataSizeUsedByte - inDataPtr;
             size_t inDataSizeUncomprVecByte = round_down_to_multiple(
                    inSizeRestByte, vector_size_byte::value
             );
             
             
            decompress_and_process_batch<
                VectorExtension,
                uncompr_f,
                intersect_processing_unit_wit,
                t_out_data_f,
                t_in_pos_r_f
            >::apply(
               inDataPtr,convert_size<uint8_t, uint64_t>(inDataSizeUncomprVecByte), witState
            );
            
            
            uint8_t * outDataAppendUncompr;
            std::tie(
                    outSizeComprByte, outDataAppendUncompr, outDataPtr
            ) = witState.m_Wit.done();
            outCountLog = witState.m_Wit.get_count_values();
           // std::cout << "finish uncompressed vectorized batch, inCount: " << convert_size<uint8_t, uint64_t>(inDataSizeUncomprVecByte) << ", outCount:" << outCountLog << "\n";
            // The size of the input column's uncompressed rest that can only
            // be processed with scalar instructions.
           const size_t inSizeScalarRemainderByte = inSizeRestByte % vector_size_byte::value;
            if(inSizeScalarRemainderByte && witState.offset < p_Data2Column->get_count_values()) {
                
               // const uint8_t * inScalar=inDataPtr + inDataSizeUncomprVecByte;
                    typename intersect_processing_unit_wit<
                        scalar<v64<uint64_t>>, uncompr_f, t_in_pos_r_f
                        >::state_t witUncomprState(
                        outDataPtr, inData2Ptr 
                       );
                    witUncomprState.offset=witState.offset;
                    witUncomprState.data2ptr=inData2Ptr;
                    witUncomprState.data2leftValues=witState.data2leftValues;
                
                    decompress_and_process_batch<
                        scalar<v64<uint64_t>>,
                        uncompr_f,
                        intersect_processing_unit_wit,
                        uncompr_f,
                        t_in_pos_r_f
                >::apply(
                        //inScalar, convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte), witUncomprState
                            inDataPtr, convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte), witUncomprState
                );
                
                
                std::tie(
                        std::ignore, std::ignore, outDataPtr
                ) = witUncomprState.m_Wit.done();
                outCountLog += witUncomprState.m_Wit.get_count_values();
                
              //  std::cout << "finish uncompressed scalar batch, inCount: " << convert_size<uint8_t, uint64_t>(inSizeScalarRemainderByte) << ", outCount:" << outCountLog << "\n";
            }
         }
         
                  
                 
        // Finish the output column.
        outDataCol->set_meta_data(
                outCountLog, outDataPtr - initOutData, outSizeComprByte
        );
         return outDataCol;
      }
};


}


#endif /* INTERSECT_COMPR_H */

