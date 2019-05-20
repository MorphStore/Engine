//
// Created by jpietrzyk on 16.05.19.
//

#ifndef MORPHSTORE_SET_UTILS_H
#define MORPHSTORE_SET_UTILS_H
enum class size_policy_set {
   ARBITRARY,
   EXPONENTIAL
};
enum class growth_policy_set {
   NONE
};

template<
   class VectorExtension,
   class BiggestSupportedVectorExtension,
   size_policy_set SPS
>
struct normalize_set {
   struct state_t {
      typename VectorExtension::vector_t m_Size;
      typename VectorExtension::vector_t m_Normalizer;
      state_t( size_t p_Size )/*:
         m_Size{set1<VectorExtension,64>(p_Size)},
         m_Normalizer{set1<VectorExtension,64>(~(BiggestSupportedVectorExtension::element_count::value-1))}{}*/
      {
         m_Size = set1<VectorExtension,64>(p_Size);
         m_Normalizer = set1<VectorExtension,64>(~(BiggestSupportedVectorExtension::element_count::value-1));
      }
   };

   template< size_policy_set U = SPS, typename std::enable_if< U == size_policy_set::ARBITRARY, int >::type = 0 >
   static typename VectorExtension::vector_t apply(
      typename VectorExtension::vector_t p_Pos,
      state_t const & p_State
   ) {
      return logical_and<VectorExtension>::apply(
         mod<VectorExtension>::apply(
            p_Pos,
            p_State.m_Size
         ),
         p_State.m_Normalizer
      );
   }
};
#endif //MORPHSTORE_SET_UTILS_H
