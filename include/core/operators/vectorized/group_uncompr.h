//
// Created by jpietrzyk on 15.04.19.
//

#ifndef MORPHSTORE_GROUP_UNCOMPR_H
#define MORPHSTORE_GROUP_UNCOMPR_H


#include <core/operators/interfaces/group.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

namespace morphstore {

template<>
const std::tuple<
   const column<uncompr_f> *,
   const column<uncompr_f> *
>
group<processing_style_t::vec256>(
   column<uncompr_f> const * const  inDataCol,
   size_t            const          outExtCountEstimate
) {
   const size_t inDataCount = inDataCol->get_count_values();
   const size_t inDataSize = inDataCol->get_size_used_byte();




}

}






#endif //MORPHSTORE_GROUP_UNCOMPR_H
