/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   rle.h
 * Author: Annett
 *
 * Created on 28. August 2019, 13:45
 */

#ifndef RLE_H
#define RLE_H

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
    
    //template<unsigned t_step>
    struct rle_f : public format {
       
        static size_t get_size_max_byte(size_t p_CountValues) {
            return convert_size<uint64_t, uint8_t>(p_CountValues)*2;
        }
        
        //static const size_t m_BlockSize = rle_l<t_bw, t_step>::m_BlockSize;
    };
 
     // ************************************************************************
    // Morph-operators (column-level)
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    
    //TODO switch for CD morph
    
    template<class t_vector_extension>
    class morph_batch_t<
            t_vector_extension,
            rle_f,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        struct state_t {
            const base_t * inBase;
            base_t * outBase;
            size_t* bytesWritten;
                        
            state_t(const base_t * p_InBase, base_t * p_OutBase) {
                inBase = p_InBase;
                outBase = p_OutBase;
                bytesWritten = NULL;
            }
        };
        
        protected:
            static inline void writeOutRunAndGoOn(base_t*& outBase, const base_t*& inBase,
                base_t& currentRunValue, const base_t*& currentRunStart,
                base_t runContinuation) {
                *outBase++ = currentRunValue;
                *outBase++ = static_cast<base_t>(inBase - currentRunStart) + runContinuation;
                currentRunStart = inBase+  runContinuation;
                currentRunValue = *currentRunStart;
            }
            
        public:
            
            static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countInBase
            ) {
                //TODO don't do this with every apply?
                vector_mask_t tableCountTrailingOnes[1 << vector_element_count::value];
                for (size_t i = 0; i < (1 << vector_element_count::value); i++) tableCountTrailingOnes[i] = __builtin_ctz(~i);
                
                const base_t* inbase = reinterpret_cast<const base_t*>(in8);
                
                const base_t* const endInbase = inbase + countInBase;
                const vector_t* const endInVec = reinterpret_cast<const vector_t*>(endInbase - vector_element_count::value);
                base_t* outbase = reinterpret_cast<base_t*>(out8);
                const base_t* const initOutbase = outbase;

                state_t s(inbase, outbase);
                
                const base_t* curbase = inbase + 1;
                base_t curRunValuebase = *inbase;
                vector_t curRunValueVec = vectorlib::set1<t_ve, vector_base_t_granularity::value>(curRunValuebase);
                const base_t* curRunStartbase = inbase;

                vector_t next;
                size_t cmp;
                
                while (reinterpret_cast<const vector_t*>(curbase) <= endInVec) {
                    next = vectorlib::load<t_ve, vectorlib::iov::UNALIGNED, vector_base_t_granularity::value>(reinterpret_cast<const base_t*>(curbase));
                    cmp = vectorlib::equal<t_ve,vector_base_t_granularity::value>::apply(curRunValueVec, next);
                    const int runContinue = tableCountTrailingOnes[cmp];
                    
                    if (runContinue == vector_element_count::value) {
                        curbase += vector_element_count::value;
                    } else {
                        writeOutRunAndGoOn(outbase, curbase, curRunValuebase, curRunStartbase, runContinue);
                        curRunValueVec = vectorlib::set1<t_ve, vector_base_t_granularity::value>(curRunValuebase);
                        curbase += runContinue;
                    }
                }

                while (curbase < endInbase) {
                    if (*curbase != curRunValuebase)
                        writeOutRunAndGoOn(outbase, curbase, curRunValuebase, curRunStartbase, 0);
                    curbase++;
                }
                writeOutRunAndGoOn(outbase, curbase, curRunValuebase, curRunStartbase, 0);

                const size_t bw = (outbase - initOutbase) * sizeof (base_t);
                if (s.bytesWritten != NULL)
                    *(s.bytesWritten) = bw;
                
                out8 = (uint8_t *)(outbase);
            }
            
            
    };
    
    template<
            class t_vector_extension
    >
    struct morph_t<
            t_vector_extension,
            rle_f,
            uncompr_f
    > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        using out_f = rle_f;
        using in_f = uncompr_f;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            
            const size_t countBase = inCol->get_count_values();
            
            auto outCol = new column<rle_f>(
                countBase * sizeof (base_t) *2
            );
            
            const uint8_t* in8 = inCol->get_data();
            
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;
            
            morph_batch<t_vector_extension,  rle_f, uncompr_f>(
                    in8, out8, countBase
            );
            
            outCol->set_meta_data(
                    countBase, out8 - initOut8 /*+ outSizeRestByte*/, out8 - initOut8
            );
            return outCol;
         }
     };
     

    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension>
    class morph_batch_t<
            t_vector_extension,
            uncompr_f,
            rle_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        struct state_t {
            const base_t * inBase;
            base_t * outBase;
            size_t* bytesWritten;
                        
            state_t(const base_t * p_InBase, base_t * p_OutBase) {
                inBase = p_InBase;
                outBase = p_OutBase;
                bytesWritten = NULL;
            }
        };
        
        public:
            
            static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countInBase
            ) {
                
                base_t* outBase = reinterpret_cast<base_t*>(out8);

                const base_t* const initOutBase = reinterpret_cast<const base_t*>(out8);
                const base_t* inBase = reinterpret_cast<const base_t*>(in8);
                
                int bits=vector_base_t_granularity::value;
                uint8_t sft=0;
                while (bits < vector_size_bit::value){
                    sft++;
                    bits*=2;
                }
                                
                state_t s(inBase, outBase);
                                
                for (size_t i = 0; i < countInBase; i += 2) {
                    const base_t value = inBase[i];
                    const base_t runlength = inBase[i + 1];
                    const size_t cnt = runlength >> sft;    
                    const size_t rest = runlength & (vector_element_count::value - 1);   
                    const vector_t valueVec = vectorlib::set1<t_ve, vector_base_t_granularity::value>(value);
                    for (size_t k = 0; k < cnt; k++) {
                        vectorlib::store<t_ve, vectorlib::iov::UNALIGNED, vector_base_t_granularity::value>(reinterpret_cast<base_t*>(outBase), valueVec);
                        outBase += vector_element_count::value;
                    }
                    for (size_t k = 0; k < rest; k++)
                        *outBase++ = value;
                }
                const size_t bw = (outBase - initOutBase) * sizeof (base_t);
                if (s.bytesWritten != NULL)
                    *(s.bytesWritten) = bw;
                
                out8 = (uint8_t *)(outBase);
            }
            
            
    };
    
    template<
            class t_vector_extension
    >
    struct morph_t<
            t_vector_extension,
            uncompr_f,
            rle_f
    > {
        IMPORT_VECTOR_BOILER_PLATE(t_vector_extension)
        using in_f = rle_f;
        using out_f = uncompr_f;
        
        static
        const column<out_f> *
        apply(const column<in_f> * inCol) {
            
            const size_t countBase = inCol->get_count_values();
            
            auto outCol = new column<uncompr_f>(
                countBase * sizeof (base_t)
            );
            
            const uint8_t* in8 = inCol->get_data();
            
            uint8_t * out8 = outCol->get_data();
            const uint8_t * const initOut8 = out8;
            
            morph_batch<t_vector_extension, uncompr_f, rle_f>(
                    in8, out8, inCol->get_size_used_byte()/sizeof(base_t)
            );
            
            outCol->set_meta_data(
                    countBase, out8 - initOut8 /*+ outSizeRestByte*/, out8 - initOut8
            );
            return outCol;
         }
     };
     
    // ************************************************************************
    // Interfaces for accessing compressed data
    // ************************************************************************
    // TODO implement the following stuff
    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    //TODO look into this
    /*
    template<
            class t_vector_extension,
          //  unsigned t_bw,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    class decompress_and_process_batch<
            t_vector_extension,
            rle_f,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
     
        };
        struct state_t {
            const base_t * inBase;
            vector_t nextOut;
                       
            state_t(const base_t * p_InBase) {
                inBase = p_InBase;
                nextOut = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);
                
            }
        };
        
         static void apply(
                const uint8_t * & in8,
                size_t countIn8,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState
        ) {
                
                const base_t* inBase = reinterpret_cast<const base_t*>(in8);
                state_t s(inBase);
                
                int bits=vector_base_t_granularity::value;
                uint8_t sft=0;
                while (bits < vector_size_bit::value){
                    sft++;
                    bits*=2;
                }
                                
                                                
                //for (size_t i = 0; i < countInBase; i += 2) {
                    
                    const base_t value = s.inBase[i];
                    const base_t runlength = s.inBase[i + 1];
                    const size_t cnt = runlength >> sft;    
                    const size_t rest = runlength & (vector_element_count::value - 1);   
                    s.nextOut = vectorlib::set1<t_ve, vector_base_t_granularity::value>(value);
                    for (size_t k = 0; k < cnt; k++) {
                        //TODO: call operator instead of storing
                        t_op_vector<t_ve, t_extra_args ...>::apply(s.nextOut, opState);
                        //vectorlib::store<t_ve, vectorlib::iov::UNALIGNED, vector_base_t_granularity::value>(reinterpret_cast<base_t*>(outBase), valueVec);
                        //outBase += vector_element_count::value;
                    }
                    s.inBase += 2;
            
                  
                //}
            
                in8 = reinterpret_cast<const uint8_t *>(s.inBase);
         }
         
    };   */
    // ------------------------------------------------------------------------
    // Random read
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Sequential write
    // ------------------------------------------------------------------------
    
     
}

#endif /* RLE_H */

