#!/usr/bin/env python3

#*********************************************************************************************
# Copyright (C) 2019 by MorphStore-Team                                                      *
#                                                                                            *
# This file is part of MorphStore - a compression aware vectorized column store.             *
#                                                                                            *
# This program is free software: you can redistribute it and/or modify it under the          *
# terms of the GNU General Public License as published by the Free Software Foundation,      *
# either version 3 of the License, or (at your option) any later version.                    *
#                                                                                            *
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
# See the GNDo this in a proper way.U General Public License for more details.                                       *
#                                                                                            *
# You should have received a copy of the GNU General Public License along with this program. *
# If not, see <http://www.gnu.org/licenses/>.                                                *
#*********************************************************************************************

"""
This script generates the C++ code of the routines for vertical bit packing
(vbp_l) for all bit widths and prints it to stdout.

In particular, it generates a partial (w.r.t. the vector extension) template
specialization of (1) the batch-level compressing morph-operator, (2) the
batch-level decompressing morph-operator, and (3) decompress_and_process_batch,
for vertical bit packing (vbp_l) for each bitwidth (from 1 to 64).
"""

# TODO The C++ code generated for these routines still has a tremendous
# potential for optimization.

import sys
print("This is vbp_routine_gen.py", file=sys.stderr)

COUNT_BITS = 64 # Since we use 64-bit integers for the uncompressed data.


# *****************************************************************************
# Utilities
# *****************************************************************************

def minimumCycleLen(bw):
    cycleLen = COUNT_BITS;
    while not (bw & 1):
        bw >>= 1
        cycleLen >>= 1
    return cycleLen;
    
def printHeader(title):
    print("// " + "-" * 70)
    print("// " + title)
    print("// " + "-" * 70)
    print()


# *****************************************************************************
# Functions generating the C++ code
# *****************************************************************************

def generateComprRoutine(bw):
    cycleLenVec = minimumCycleLen(bw)

    print("template<class t_vector_extension>")
    print("class morph_batch_t<")
    print("        t_vector_extension,")
    print("        vbp_l<{}, t_vector_extension::vector_helper_t::element_count::value>,".format(bw))
    print("        uncompr_f")
    print("> {")
    print("    using t_ve = t_vector_extension;")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print()
    print("    static const unsigned m_Bw = {};".format(bw))
    print()
    print("    using dst_l = vbp_l<m_Bw, t_vector_extension::vector_helper_t::element_count::value>;")
    print()
    print("public:")
    print("#ifdef VBP_FORCE_INLINE_PACK")
    print("    MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("    static void apply(")
    print("            const uint8_t * & in8, uint8_t * & out8, size_t countInLog")
    print("    ) {")
    print("        using namespace vectorlib;")
    print()
    print("        vector_t tmp = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);")
    print("        vector_t tmp2;")
    print()
    print("        const base_t * inBase = reinterpret_cast<const base_t *>(in8);")
    print("        base_t * outBase = reinterpret_cast<base_t *>(out8);")
    print()
    print("        const size_t cycleLenVec = {};".format(cycleLenVec))
    print("        const size_t cycleLenBase = cycleLenVec * vector_element_count::value;")
    print("        const size_t cycleCount = countInLog / cycleLenBase;")
    print("        for(size_t i = 0; i < cycleCount; i++) {")

    bitpos = 0
    for posInCycle in range(0, cycleLenVec):
        print("            tmp2 = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inBase);")
        print("            inBase += vector_element_count::value;")
        print("            tmp = bitwise_or<t_ve>(tmp, shift_left<t_ve>::apply(tmp2, {}));".format(bitpos))
        bitpos += bw
        if (posInCycle + 1) * bw % COUNT_BITS == 0:
            print("            store<t_ve, iov::ALIGNED, vector_size_bit::value>(outBase, tmp);")
            print("            outBase += vector_element_count::value;")
            print("            tmp = set1<t_ve, vector_base_t_granularity::value>(0);")
            bitpos = 0
        elif(int(posInCycle * bw / COUNT_BITS) < int(((posInCycle + 1) * bw - 1) / COUNT_BITS)):
            print("            store<t_ve, iov::ALIGNED, vector_size_bit::value>(outBase, tmp);")
            print("            outBase += vector_element_count::value;")
            print("            tmp = shift_right<t_ve>::apply(tmp2, {});".format(bw - bitpos + COUNT_BITS))
            bitpos -= COUNT_BITS

    print("        }")
    print()
    print("        in8 = reinterpret_cast<const uint8_t *>(inBase);")
    print("        out8 = reinterpret_cast<uint8_t *>(outBase);")
    print("    }")
    print("};")
    
def generateDecomprRoutine(bw):
    cycleLenVec = minimumCycleLen(bw)

    print("template<class t_vector_extension>")
    print("class morph_batch_t<")
    print("        t_vector_extension,")
    print("        uncompr_f,")
    print("        vbp_l<{}, t_vector_extension::vector_helper_t::element_count::value>".format(bw))
    print("> {")
    print("    using t_ve = t_vector_extension;")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print()
    print("    static const unsigned m_Bw = {};".format(bw))
    print()
    print("    using src_l = vbp_l<m_Bw, t_vector_extension::vector_helper_t::element_count::value>;")
    print()
    print("public:")
    print("#ifdef VBP_FORCE_INLINE_UNPACK")
    print("    MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("    static void apply(")
    print("            const uint8_t * & in8, uint8_t * & out8, size_t countLog")
    print("    ) {")
    print("        using namespace vectorlib;")
    print()
    print("        const vector_t mask = set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(m_Bw));")
    print("        vector_t tmp = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);")
    print("        vector_t nextOut = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);")
    print()
    print("        const base_t * inBase = reinterpret_cast<const base_t *>(in8);")
    print("        base_t * outBase = reinterpret_cast<base_t *>(out8);")
    print()
    print("        const size_t cycleLenVec = {};".format(cycleLenVec))
    print("        const size_t cycleLenBase =")
    print("                cycleLenVec * vector_element_count::value;")
    print("        for(size_t i = 0; i < countLog; i += cycleLenBase) {")

    bitpos = 0
    for posInCycle in range(0, cycleLenVec):
        if (posInCycle * bw) % COUNT_BITS == 0:
            print("            tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inBase);")
            print("            inBase += vector_element_count::value;")
            print("            nextOut = bitwise_and<t_ve>(mask, tmp);")
            bitpos = bw
        elif int(posInCycle * bw / COUNT_BITS) < int(((posInCycle + 1) * bw - 1) / COUNT_BITS):
            print("            tmp = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inBase);")
            print("            inBase += vector_element_count::value;")
            print("            nextOut = bitwise_and<t_ve>(mask, bitwise_or<t_ve>(shift_left<t_ve>::apply(tmp, {}), nextOut));".format(COUNT_BITS - bitpos + bw))
            bitpos -= COUNT_BITS
        print("            store<t_ve, iov::ALIGNED, vector_size_bit::value>(outBase, nextOut);")
        print("            outBase += vector_element_count::value;")
        print("            nextOut = bitwise_and<t_ve>(mask, shift_right<t_ve>::apply(tmp, {}));".format(bitpos))
        bitpos += bw;

    print("        }")
    print()
    print("        in8 = reinterpret_cast<const uint8_t *>(inBase);")
    print("        out8 = reinterpret_cast<uint8_t *>(outBase);")
    print("    }")
    print("};")
    
def generateDecomprAndProcessRoutine(bw, factor):
    cycleLenVec = minimumCycleLen(bw)

    print("template<class t_vector_extension, template<class, class ...> class t_op_vector, class ... t_extra_args>")
    print("class decompress_and_process_batch<")
    print("        t_vector_extension,")
    print("        vbp_l<{}, t_vector_extension::vector_helper_t::element_count::value * {}>,".format(bw, factor))
    print("        t_op_vector,")
    print("        t_extra_args ...")
    print("> {")
    print("    using t_ve = t_vector_extension;")
    print("    IMPORT_VECTOR_BOILER_PLATE(t_ve)")
    print()
    print("    static const unsigned m_Bw = {};".format(bw))
    print()
    print("    using src_l = vbp_l<m_Bw, t_vector_extension::vector_helper_t::element_count::value * {}>;".format(factor))
    print()
    print("public:")
    print("#ifdef VBP_FORCE_INLINE_UNPACK")
    print("    MSV_CXX_ATTRIBUTE_FORCE_INLINE")
    print("#endif")
    print("    static void apply(")
    print("            const uint8_t * & in8, size_t countInLog, typename t_op_vector<t_ve, t_extra_args ...>::state_t & opState")
    print("    ) {")
    print("        using namespace vectorlib;")
    print()
    print("        const vector_t mask = set1<t_ve, vector_base_t_granularity::value>(bitwidth_max<base_t>(m_Bw));")
    for i in range(factor):
        print("        vector_t tmp{} = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);".format(i))
        print("        vector_t nextOut{} = vectorlib::set1<t_ve, vector_base_t_granularity::value>(0);".format(i))
    print()
    print("        const base_t * inBase = reinterpret_cast<const base_t *>(in8);")
    print()
    print("        const size_t cycleLenVec = {};".format(cycleLenVec))
    print("        const size_t cycleLenBase =")
    print("                cycleLenVec * vector_element_count::value;")
    print("        for(size_t i = 0; i < countInLog; i += cycleLenBase * {}) {{".format(factor))

    bitpos = 0
    for posInCycle in range(0, cycleLenVec):
        if (posInCycle * bw) % COUNT_BITS == 0:
            for i in range(factor):
                print("            tmp{} = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inBase);".format(i))
                print("            inBase += vector_element_count::value;")
                print("            nextOut{} = bitwise_and<t_ve>(mask, tmp{});".format(i, i))
            bitpos = bw
        elif int(posInCycle * bw / COUNT_BITS) < int(((posInCycle + 1) * bw - 1) / COUNT_BITS):
            for i in range(factor):
                print("            tmp{} = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inBase);".format(i))
                print("            inBase += vector_element_count::value;")
                print("            nextOut{} = bitwise_and<t_ve>(mask, bitwise_or<t_ve>(shift_left<t_ve>::apply(tmp{}, {}), nextOut{}));".format(i, i, COUNT_BITS - bitpos + bw, i))
            bitpos -= COUNT_BITS
        for i in range(factor):
            print("            t_op_vector<t_ve, t_extra_args ...>::apply(nextOut{}, opState);".format(i))
            print("            nextOut{} = bitwise_and<t_ve>(mask, shift_right<t_ve>::apply(tmp{}, {}));".format(i, i, bitpos))
        bitpos += bw;

    print("        }")
    print()
    print("        in8 = reinterpret_cast<const uint8_t *>(inBase);")
    print("    }")
    print("};")
    
    
# *****************************************************************************
# Main program
# *****************************************************************************
    
if __name__ == "__main__":
    print("// This file was automatically generated by vbp_routine_gen.py")
    print()
    print("namespace morphstore {")
    print()
    
    printHeader("Compression")
    for bw in range(1, COUNT_BITS+1):
        generateComprRoutine(bw)
        print()
        print()

    printHeader("Decompression")
    for bw in range(1, COUNT_BITS + 1):
        generateDecomprRoutine(bw)
        print()
        print()

    printHeader("Decompression and processing")
    for bw in range(1, COUNT_BITS + 1):
        generateDecomprAndProcessRoutine(bw, 1)
        generateDecomprAndProcessRoutine(bw, 2)
        generateDecomprAndProcessRoutine(bw, 4)
        generateDecomprAndProcessRoutine(bw, 8)
        print()
        print()
        
    print("}")