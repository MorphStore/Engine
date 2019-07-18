//
// Created by jpietrzyk on 17.07.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_VECTORIZED_GROUP_NEC_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_VECTORIZED_GROUP_NEC_UNCOMPR_H

namespace morphstore {

   static
   const std::tuple<
      const column<uncompr_f> *,
      const column<uncompr_f> *
   >
   apply(//         size_t            m_ResultCount;
      column<uncompr_f> const * const p_InDataCol,
      size_t const outCountEstimate = 0
   ) {

      _ve_lvl(256);
      __vr const hashMultiplyVec = _ve_vbrd_vs_i64( ( ( 1 << 16 ) + 1 ) );
      __vr const resizeVec = _ve_vbrd_vs_i64( mapSize - 1 );
//      __vr const alignerVec = _ve_vbrd_vs_i64( 255 );

      __vm256 activeLaneMask = ...; //SET ALL TO ONE

      //START LOOP

         //Load
         __vr dataVec = _ve_vld_vss( sizeof( T ), p_DataPtr );
         //Hash START
            __vr hashVec = _ve_vmulul_vvv( data, hashMultiplyVec );
            //Resize
            __vr resizedHashVec = _ve_vand_vvv( hashVec, resizeVec );
            //Align
            //__vr alignedHashVec = _ve_vand_vv( resizedHashVec, alignerVec );
         //Hash END
         //Gathering from Map
         __vr gatheredVec = _ve_vgt_vvm(
               _ve_vgt_vv( _ve_vsfa_vvss( resizedHashVec, 3, ( unsigned long int ) hashMapPtr ) ),
               activeLaneMask
         );
         //Comparing for Equal @todo: SHOULD BE NOT EQUAL
         __vm256 equalsMask = _ve_vfmkl_mcv( VECC_EQ, _ve_vcmpsl_vvv( dataVec, gatheredVec ) );

         //Comparing for Zero
         __vm256 equalsZero = _ve_vfmkl_mcv( VECC_EQ, _ve_vcmpsl_vvv( zeroVec, gatheredVec ) );






   }


}
#endif //MORPHSTORE_CORE_OPERATORS_VECTORIZED_GROUP_NEC_UNCOMPR_H
