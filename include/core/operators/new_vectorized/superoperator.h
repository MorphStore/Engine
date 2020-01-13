
#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>



namespace morphstore {

    using namespace vectorlib;

    template< template<class, class> class Operator, class VectorExtension, class InFormatCol>
    struct superoperator;


    template < template<class, class> class Operator, class VectorExtension> 
    struct superoperator <Operator, VectorExtension, uncompr_f> {
        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)


        // Fall 1: Unärer Operator, agg_sum, unary_calc, 
        // @todo noch nicht allgemein, nur für agg_sum
        MSV_CXX_ATTRIBUTE_FORCE_INLINE static
        const column<uncompr_f> *
        apply(column< uncompr_f > const * const p_DataColumn) {
           

            size_t const vectorCount = p_DataColumn->get_count_values() / vector_element_count::value;
            size_t const remainderCount = p_DataColumn->get_count_values() % vector_element_count::value;
            base_t const * dataPtr = p_DataColumn->get_data( );



         typename Operator<VectorExtension, uncompr_f>::state_t vectorState;
         for(size_t i = 0; i < vectorCount; ++i) {
            vector_t dataVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(dataPtr);
            Operator<VectorExtension, uncompr_f>::apply(
               dataVector,
               vectorState
            );
            dataPtr += vector_element_count::value;
         }
        static base_t t = Operator<VectorExtension, uncompr_f>::finalize( vectorState );
        typename Operator<scalar<v64<uint64_t>>, uncompr_f>::state_t scalarState(t);

        for(size_t i = 0; i < remainderCount; ++i) {
            vector_t dataVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(dataPtr);
            Operator<VectorExtension, uncompr_f>::apply(
               dataVector,
               scalarState
            );
            dataPtr += vector_element_count::value;
         }

        base_t result = Operator<VectorExtension, uncompr_f>::finalize( scalarState );
        auto outDataCol = new column<uncompr_f>(sizeof(base_t));
        base_t * const outData = outDataCol->get_data();
        *outData = result;
        outDataCol->set_meta_data(1, sizeof(base_t));
        return outDataCol;


        }


        // // Fall 2: Unärer Operator, Rückgabe Tupel, Group1
        // MSV_CXX_ATTRIBUTE_FORCE_INLINE 
        // static
        // const std::tuple<
        // const column<uncompr_f> *,
        // const column<uncompr_f> *
        // >
        // apply2(         
        //     column< uncompr_f > const * const p_Data1Column,
        //     const size_t p_OutPosCountEstimate = 0) {
           
        // }


        // // Fall 3: Binärer Operator, binary_calc, intersect, merge, project
        // MSV_CXX_ATTRIBUTE_FORCE_INLINE static
        // const column<uncompr_f> *
        // apply(         
        //     column< uncompr_f > const * const p_Data1Column,
        //     column< uncompr_f > const * const p_Data2Column,
        //     const size_t p_OutPosCountEstimate = 0) {
           

        // }

        // // Fall 4: Binärer Operator, Rückgabe Tupel, Group2, Join
        // MSV_CXX_ATTRIBUTE_FORCE_INLINE 
        // static
        // const std::tuple<
        // const column<uncompr_f> *,
        // const column<uncompr_f> *
        // >
        // apply2(         
        //     column< uncompr_f > const * const p_Data1Column,
        //     column< uncompr_f > const * const p_Data2Column,
        //     const size_t p_OutPosCountEstimate = 0) {
           

        // }



        // // Fall 5: Binärer Operator mit Prädikat, select
        // MSV_CXX_ATTRIBUTE_FORCE_INLINE 
        // static
        // const column<uncompr_f> *
        // apply2(         
        // column< uncompr_f > const * const p_DataColumn,
        // typename t_vector_extension::vector_helper_t::base_t const p_Predicate,
        // const size_t outPosCountEstimate = 0) {
           

        // }
    };

}