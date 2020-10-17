#!/usr/bin/env python3

"""
This script generates the C++ code of the routines for the Group-Simple
algorithm (group_simple_f), i.e. special vertical bit packing routines, for all
possible compression modes of that algorithm and prints it to stdout.

This script assumes that the base type is uint64_t.
"""

import sys
print("This is group_simple_routine_gen.py", file=sys.stderr)

COUNT_BITS = 64 # Since we use 64-bit integers for the uncompressed data.


# *****************************************************************************
# Utilities
# *****************************************************************************

def printHeader(title):
    print("// " + "-" * 70)
    print("// " + title)
    print("// " + "-" * 70)
    print()


# *****************************************************************************
# Functions generating the C++ code
# *****************************************************************************

def generateComprSwitch(bws):
    print("template<class t_ve>")
    print("#ifdef GROUPSIMPLE_FORCE_INLINE_PACK_SWITCH")
    print("MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("static void compress_complete_block(unsigned p_Sel, const typename t_ve::base_t * & p_InBase, typename t_ve::base_t * & p_OutBase) {")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print("    using namespace vectorlib;")
    print("    vector_t res;")
    print("    // In the following, n means the number of groups and b means the bit width.")
    print("    switch(p_Sel) {")
    for idx, bw in enumerate(bws):
        n = int(COUNT_BITS / bw)
        print("        case {:> 2}: // n = {:> 2}, b = {:> 2}".format(idx, n, bw))
        print("            res = unrolled_packing_{}_{}x{}bit<t_ve>(p_InBase); break;".format(COUNT_BITS, n, bw))
    print("    }")
    print("    store<t_ve, iov::ALIGNED, vector_size_bit::value>(p_OutBase, res);")
    print("    p_OutBase += vector_element_count::value;")
    print("}")
    
def generateDecomprSwitch(bws):
    print("template<class t_ve>")
    print("#ifdef GROUPSIMPLE_FORCE_INLINE_UNPACK_SWITCH")
    print("MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("static void decompress_complete_block(unsigned p_Sel, const typename t_ve::base_t * & p_InBase, typename t_ve::base_t * & p_OutBase) {")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print("    using namespace vectorlib;")
    print("    const vector_t comprBlock = load<t_ve, iov::ALIGNED, vector_size_bit::value>(p_InBase);")
    print("    p_InBase += vector_element_count::value;")
    print("    // In the following, n means the number of groups and b means the bit width.")
    print("    switch(p_Sel) {")
    for idx, bw in enumerate(bws):
        n = int(COUNT_BITS / bw)
        print("        case {:> 2}: // n = {:> 2}, b = {:> 2}".format(idx, n, bw))
        print("            unrolled_unpacking_{}_{}x{}bit<t_ve>(comprBlock, p_OutBase); break;".format(COUNT_BITS, n, bw))
    print("    }")
    print("}")
    
def generateDecomprAndProcessSwitch(bws):
    print("template<")
    print("    class t_ve,")
    print("    template<class, class ...> class t_op_vector,")
    print("    class ... t_extra_args")
    print(">")
    print("#ifdef GROUPSIMPLE_FORCE_INLINE_UNPACK_AND_PROCESS_SWITCH")
    print("MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("static void decompress_and_process_complete_block(")
    print("        unsigned p_Sel,")
    print("        const typename t_ve::base_t * & p_InBase,")
    print("        typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State")
    print(") {")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print("    using namespace vectorlib;")
    print("    const vector_t comprBlock = load<t_ve, iov::ALIGNED, vector_size_bit::value>(p_InBase);")
    print("    p_InBase += vector_element_count::value;")
    print("    // In the following, n means the number of groups and b means the bit width.")
    print("    switch(p_Sel) {")
    for idx, bw in enumerate(bws):
        n = int(COUNT_BITS / bw)
        print("        case {:> 2}: // n = {:> 2}, b = {:> 2}".format(idx, n, bw))
        print("            unrolled_unpacking_and_processing_{}_{}x{}bit<t_ve, t_op_vector, t_extra_args ...>(comprBlock, p_State); break;".format(COUNT_BITS, n, bw))
    print("    }")
    print("}")

def generateComprRoutine(bw):
    countGroups = int(COUNT_BITS / bw)
    
    print("template<class t_ve>")
    print("#ifdef GROUPSIMPLE_FORCE_INLINE_PACK")
    print("MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("typename t_ve::vector_t unrolled_packing_{}_{}x{}bit(const typename t_ve::base_t * & in) {{".format(COUNT_BITS, countGroups, bw))
    print("    using namespace vectorlib;")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print("    vector_t res = load<t_ve, iov::ALIGNED, vector_size_bit::value>(in);")
    print("    in += vector_element_count::value;")
    for i in range(1, countGroups):
        print("    res = bitwise_or<t_ve, vector_size_bit::value>(res, shift_left<t_ve, vector_base_t_granularity::value>::apply(load<t_ve, iov::ALIGNED, vector_size_bit::value>(in), {}));".format(i * bw))
        print("    in += vector_element_count::value;")
    print("    return res;")
    print("}")

def generateDecomprRoutine(bw):
    countGroups = int(COUNT_BITS / bw)
    
    print("template<class t_ve>")
    print("#ifdef GROUPSIMPLE_FORCE_INLINE_UNPACK")
    print("MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("void unrolled_unpacking_{}_{}x{}bit(".format(COUNT_BITS, countGroups, bw))
    print("        const typename t_ve::vector_t & comprBlock,")
    print("        typename t_ve::base_t * & out")
    print(") {")
    print("    using namespace vectorlib;")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    if countGroups == 1: # 64 bit per data element (no compression)
        # Directly copy the input to the output.
        print("    store<t_ve, iov::ALIGNED, vector_size_bit::value>(out, comprBlock);")
        print("    out += vector_element_count::value;")
    else: # < 64 bit per data element (actual compression) 
        # Decompress the data elements.
        print("    const vector_t mask = set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>({}));".format(bw))
        # The first group does not need to be shifted.
        print("    store<t_ve, iov::ALIGNED, vector_size_bit::value>(out, bitwise_and<t_ve, vector_size_bit::value>(comprBlock, mask));")
        print("    out += vector_element_count::value;")
        # All inner groups need to be shifted and masked.
        for i in range(1, countGroups - 1):
            print("    store<t_ve, iov::ALIGNED, vector_size_bit::value>(out, bitwise_and<t_ve, vector_size_bit::value>(shift_right<t_ve, vector_base_t_granularity::value>::apply(comprBlock, {}), mask));".format(i * bw))
            print("    out += vector_element_count::value;")
        # The last group does not need to be masked.
        print("    store<t_ve, iov::ALIGNED, vector_size_bit::value>(out, shift_right<t_ve, vector_base_t_granularity::value>::apply(comprBlock, {}));".format((countGroups - 1) * bw))
        print("    out += vector_element_count::value;")
    print("}")

def generateDecomprAndProcessRoutine(bw):
    countGroups = int(COUNT_BITS / bw)
    
    print("template<")
    print("    class t_ve,")
    print("    template<class, class ...> class t_op_vector,")
    print("    class ... t_extra_args")
    print(">")
    print("#ifdef GROUPSIMPLE_FORCE_INLINE_UNPACK_AND_PROCESS")
    print("MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("void unrolled_unpacking_and_processing_{}_{}x{}bit(".format(COUNT_BITS, countGroups, bw))
    print("        const typename t_ve::vector_t & comprBlock,")
    print("        typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State")
    print(") {")
    print("    using namespace vectorlib;")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    if countGroups == 1: # 64 bit per data element (no compression)
        # Directly forward the input to the operator core.
        print("    t_op_vector<t_ve, t_extra_args ...>::apply(comprBlock, p_State);")
    else: # < 64 bit per data element (actual compression) 
        print("    const vector_t mask = set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>({}));".format(bw))
        # The first group does not need to be shifted.
        print("    t_op_vector<t_ve, t_extra_args ...>::apply(bitwise_and<t_ve, vector_size_bit::value>(comprBlock, mask), p_State);")
        # All inner groups need to be shifted and masked.
        for i in range(1, countGroups - 1):
            print("    t_op_vector<t_ve, t_extra_args ...>::apply(bitwise_and<t_ve, vector_size_bit::value>(shift_right<t_ve, vector_base_t_granularity::value>::apply(comprBlock, {}), mask), p_State);".format(i * bw))
        # The last group does not need to be masked.
        print("    t_op_vector<t_ve, t_extra_args ...>::apply(shift_right<t_ve, vector_base_t_granularity::value>::apply(comprBlock, {}), p_State);".format((countGroups - 1) * bw))
    print("}")
    
    
# *****************************************************************************
# Main program
# *****************************************************************************
    
if __name__ == "__main__":
    guard = "MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_ROUTINES_H"
    
    print("// This file was automatically generated by group_simple_routine_gen.py .")
    print("// It should not be committed.")
    print()
    print("#ifndef {}".format(guard))
    print("#define {}".format(guard))
    print()
    print("#include <core/morphing/group_simple_commons.h>")
    print()
    print("namespace morphstore {")
    print("namespace group_simple_routines {")
    print()
    
    # This is for a 64-bit base type.
    bws = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 21, 32, 64]
    
    printHeader("Compression")
    for bw in bws:
        generateComprRoutine(bw)
        print()
        print()
    generateComprSwitch(bws)

    printHeader("Decompression")
    for bw in bws:
        generateDecomprRoutine(bw)
        print()
        print()
    generateDecomprSwitch(bws)

    printHeader("Decompression and processing")
    for bw in bws:
        generateDecomprAndProcessRoutine(bw)
        print()
        print()
    generateDecomprAndProcessSwitch(bws)
        
    print()
    print("} // namespace group_simple_routines")
    print("} // namespace morphstore")
    print()
    print("#endif //{}".format(guard))