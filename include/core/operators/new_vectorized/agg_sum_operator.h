#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

namespace morphstore {


    using namespace vectorlib;

    template <class VectorExtension, class InFormatCol>
    struct agg_sum_operator;


    template <class VectorExtension>
    struct agg_sum_operator <VectorExtension, uncompr_f>  {

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_t resultVec;
         state_t(void): resultVec( vectorlib::set1<VectorExtension, vector_base_t_granularity::value>( 0 ) ) { }
         //state_t(vector_t const & p_Data): resultVec( p_Data ) { }
         state_t(base_t p_Data): resultVec(vectorlib::set1<scalar<v64<uint64_t>>,64>(p_Data)){}
         //TODO replace by set
      };
      
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         vector_t const & p_DataVector,
         state_t & p_State
         // ,size_t element_count = vector_element_count::value
      ) {
         p_State.resultVec = vectorlib::add<VectorExtension, vector_base_t_granularity::value>::apply(
            p_State.resultVec, p_DataVector
            // ,element_count
         );
      }
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static base_t finalize(
         typename agg_sum_operator<VectorExtension, uncompr_f>::state_t & p_State
      ) {
          
         return vectorlib::hadd<VectorExtension,vector_base_t_granularity::value>::apply( p_State.resultVec );
      }
            

    };

}