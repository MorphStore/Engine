/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/


#ifndef QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_OPERATORS_UNCOMPR_AGG_SUM_MERGE_H
#define QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_OPERATORS_UNCOMPR_AGG_SUM_MERGE_H

#include <core/operators/interfaces/agg_sum_merge.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {
    template<
      class TVectorExtension
    >
    struct agg_sum_merge_t<TVectorExtension, uncompr_f, uncompr_f, uncompr_f, uncompr_f> {
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        const std::tuple<
          const column <uncompr_f> *,
          const column <uncompr_f> *
        >
        static
        apply(column<uncompr_f> const * const * const in_groups,
              column<uncompr_f> const * const * const in_sums,
              const uint64_t partitions
        ){
            unordered_map<uint64_t, uint64_t> mergedAggr;
//            column<uncompr_f> const * inGroupCol;
//            column<uncompr_f> const * inPartAggSumCol;
            
            const uint64_t * inGroup;
            const uint64_t * inPartAggSum;
            
            size_t inDataCount = 0;
            unordered_map<uint64_t, uint64_t>::iterator found;
            
        //	for (uint64_t colNr = 0; colNr < inColArraySize; ++colNr) {
        //		debug(colNr << ": " << inCharColArray[colNr]->getMeta()->cid << " / " << inPartAggColArray[colNr]->getMeta()->cid);
        //	}
            for (uint64_t colNr = 0; colNr < partitions; ++colNr) {
                inGroup = in_groups[colNr]->get_data();
                inPartAggSum = in_sums[colNr]->get_data();
                
                
                inDataCount = in_groups[colNr]->get_count_values();
                
                for(uint64_t colPos = 0; colPos < inDataCount; ++colPos){
                    found = mergedAggr.find(inGroup[colPos]);
                    if(found == mergedAggr.end()){
                        mergedAggr[inGroup[colPos]] = inPartAggSum[colPos];
                    } else {
                        mergedAggr[inGroup[colPos]] += inPartAggSum[colPos];
                    }
                }
            }
            
            column<uncompr_f> * outChaCol = new column<uncompr_f>(mergedAggr.size() * sizeof(uint64_t));
            column<uncompr_f> * outAggCol = new column<uncompr_f>(mergedAggr.size() * sizeof(uint64_t));
            
            uint64_t * outCha = outChaCol->get_data();
            uint64_t * outAgg = outAggCol->get_data();
            
            size_t pos = 0;
            
            for(auto& pair : mergedAggr){
                outCha[pos] = pair.first;
                outAgg[pos] = pair.second;
                ++pos;
            }
            
            outChaCol->set_meta_data(mergedAggr.size(), mergedAggr.size()*sizeof(uint64_t));
            outAggCol->set_meta_data(mergedAggr.size(), mergedAggr.size()*sizeof(uint64_t));
            return std::make_tuple(outChaCol, outAggCol);

        }
        
    };
} /// namespace morphstore

#endif //QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_OPERATORS_UNCOMPR_AGG_SUM_MERGE_H
