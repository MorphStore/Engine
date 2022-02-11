/**
 * @file group_simple_commons.h
 * @brief Some central macros for the format `group_simple_f`.
 */

#ifndef MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_COMMONS_H
#define MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_COMMONS_H

/**
 * The following macros control whether the respective kind of routine is
 * declared to be always inlined.
 * 
 * These hardcoded decisions are based on a small experiment which showed that
 * especially inlining the switching routines is very beneficial, while also
 * inlining the unpacking routines yields a little additional improvement. This
 * might be different depending on the hardware, though.
 */
#undef GROUPSIMPLE_FORCE_INLINE_PACK
#define GROUPSIMPLE_FORCE_INLINE_PACK_SWITCH
#define GROUPSIMPLE_FORCE_INLINE_UNPACK
#define GROUPSIMPLE_FORCE_INLINE_UNPACK_SWITCH
#define GROUPSIMPLE_FORCE_INLINE_UNPACK_AND_PROCESS
#define GROUPSIMPLE_FORCE_INLINE_UNPACK_AND_PROCESS_SWITCH

#endif //MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_COMMONS_H
