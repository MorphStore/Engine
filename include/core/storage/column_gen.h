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

/**
 * @file column_gen.h
 * @brief A collection of functions for creating uncompressed columns and
 * initializing them with synthetically generated data.
 */

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_GEN_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_GEN_H

#include <core/storage/column.h>
#include <core/morphing/format.h>
#include <core/utils/basic_types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace morphstore {
    
    
    class generate_with_bitwidth_histogram_helpers {
      public:
        /**
         * @brief Reads bit width weights to be used with
         * `generate_with_bitwidth_histogram` from a binary file.
         *
         * The file contents must be a concatenation of any number of 64-byte
         * units, each of which represents one bit width histogram. Within each
         * unit, the i-th byte is the weight of bit width i in the bit width
         * histogram.
         *
         * For the in-memory representation, we use a uint64_t for each weight.
         * Thus, the file contents are expanded.
         *
         * @param p_BwWeightsFilePath The path to the file containing the bit width
         * weights.
         * @return A pair of (1) a pointer to the buffer containing the weights and
         * (2) the number of histograms in that buffer.
         * @todo The file layout should not be limited to one byte per weight.
         */
        static
        std::pair<uint64_t *, size_t>
        read_bw_weights(const std::string & p_BwWeightsFilePath) {
            std::ifstream ifs(
                    p_BwWeightsFilePath, std::ios::in | std::ios::binary
            );
            if(ifs.good()) {
                ifs.seekg(0, std::ios_base::end);
                const size_t sizeByte = ifs.tellg();
                if(sizeByte % std::numeric_limits<uint64_t>::digits)
                    throw std::runtime_error(
                            "the number of weights must be a multiple of 64"
                    );
                ifs.seekg(0, std::ios_base::beg);
    
                uint8_t * wsBufTmp = new uint8_t[sizeByte];
                ifs.read(reinterpret_cast<char*>(wsBufTmp), sizeByte);
                if(!ifs.good())
                    throw std::runtime_error("could not read weights data from file");
    
                // As many uint64_t as there are bytes in the file.
                uint64_t * wsBuf = new uint64_t[sizeByte];
                for(size_t i = 0; i < sizeByte; i++)
                    wsBuf[i] = wsBufTmp[i];
    
                delete[] wsBufTmp;
    
                return std::make_pair(
                        wsBuf, sizeByte / std::numeric_limits<uint64_t>::digits
                );
            }
            else
                throw std::runtime_error("could not open weights file for reading");
        }
        
        MSV_CXX_ATTRIBUTE_INLINE
        static
        uint64_t
        get_value(uint64_t x, unsigned bwMinus1) {
            return (
                (
                    // random 64-bit number
                    x
                    // ensure that there are at most (bw-1) effective bits
                    & ~(std::numeric_limits<uint64_t>::max() << bwMinus1)
                )
                // ensure that there are at least bw effective bits
                | (static_cast<uint64_t>(1) << bwMinus1)
                // ensured that there are exactly bw effective bits
            );
        }
    };

    

    class ColumnGenerator {
      public:
        /**
         * @brief Creates an uncompressed column and copies the contents of the given
         * vector into that column's data buffer. This is a convenience function for
         * creating small toy example columns. To prevent its usage for non-toy
         * examples, it throws an exception if the given vector contains more than 20
         * elements.
         *
         * @param vec The vector to initialize the column with.
         * @return An uncompressed column containing a copy of the data in the given
         * vector.
         */
        static const column <uncompr_f> * make_column(const std::vector<uint64_t> & vec) {
            const size_t count = vec.size();
            if (count > 20)
                throw std::runtime_error(
                  "make_column() is an inefficient convenience function and "
                  "should only be used for very small columns"
                );
            const size_t size = count * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(size);
            memcpy(resCol->get_data(), vec.data(), size);
            resCol->set_meta_data(count, size);
            return resCol;
        }
    
    
        const column <uncompr_f> * make_column(uint64_t const * const vec, size_t count) {
            if (count > 400)
                throw std::runtime_error(
                  "make_column() is an inefficient convenience function and "
                  "should only be used for very small columns"
                );
            const size_t size = count * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(size);
            memcpy(resCol->get_data(), vec, size);
            resCol->set_meta_data(count, size);
            return resCol;
        }
    
        /**
         * @brief Creates an uncompressed column and fills its data buffer with sorted
         * unique data elements according to an arithmetic sequence. Can be used to
         * generate primary key columns.
         *
         * @param countValues The number of data elements to generate.
         * @param start The first data element.
         * @param step The difference between two consecutive data elements.
         * @return A column whose i-th data element is start + i * step .
         */
        static
        const column <uncompr_f> *
        generate_sorted_unique(size_t countValues, uint64_t start = 0, uint64_t step = 1) {
            const size_t allocationSize = countValues * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(allocationSize);
            uint64_t * const res = resCol->get_data();
        
            for (unsigned i = 0; i < countValues; i ++)
                res[i] = start + i * step;
        
            resCol->set_meta_data(countValues, allocationSize);
        
            return resCol;
        }
    
    
        /**
         * @brief Creates an uncompressed column and fills its data buffer with sorted
         * unique data elements extracted uniformly from some population. Can be used
         * to generate position columns like those output by selective query operators.
         *
         * @param p_CountValues The number of data elements to generate, i.e., to draw
         * from the population.
         * @param p_CountPopulation The size of the underlying population, must not be
         * less than p_CountValues.
         * @return A sorted column containing p_CountValues uniques data elements from
         * the range [0, p_CountPopulation - 1].
         */
        static
        const column <uncompr_f> *
        generate_sorted_unique_extraction(size_t p_CountValues, size_t p_CountPopulation, size_t p_Seed = 0) {
            const size_t allocationSize = p_CountValues * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(allocationSize);
            uint64_t * res = resCol->get_data();
        
            if (p_Seed == 0)
                p_Seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine gen(p_Seed);
            std::uniform_int_distribution<uint64_t> distr(0, p_CountPopulation - 1);
        
            // If we want to select at most half of the population, then we start with
            // no values selected and select values until we have the specified number
            // of unique values. If we want to select more than half of the population,
            // then we start with all values selected and unselect values until we have
            // the specified number.
            bool flip = p_CountValues > p_CountPopulation / 2;
            // If i-th bit is set, then i shall be output.
            std::vector<bool> chosen(p_CountPopulation, flip);
            if (! flip) { // Select values.
                size_t countChosen = 0;
                while (countChosen < p_CountValues) {
                    const uint64_t val = distr(gen);
                    if (! chosen[val]) {
                        chosen[val] = true;
                        countChosen ++;
                    }
                }
            } else { // Unselect values.
                size_t countUnChosen = 0;
                while (countUnChosen < p_CountPopulation - p_CountValues) {
                    const uint64_t val = distr(gen);
                    if (chosen[val]) {
                        chosen[val] = false;
                        countUnChosen ++;
                    }
                }
            }
        
            for (size_t i = 0; i < p_CountPopulation; i ++)
                if (chosen[i])
                    *res ++ = i;
        
            resCol->set_meta_data(p_CountValues, allocationSize);
        
            return resCol;
        }
    
    
        /**
         * @brief Creates an uncompressed column and fills its data buffer with values
         * drawn from the given random distribution. Suitable distributions can be
         * found in the STL's `<random>` header. In particular, the following
         * distributions are supported:
         * - `std::uniform_int_distribution`
         * - `std::binomial_distribution`
         * - `std::geometric_distribution`
         * - `std::negative_binomial_distribution`
         * - `std::poisson_distribution`
         * - `std::discrete_distribution`
         * Optionally, the generated data can be sorted as an additional step.
         *
         * @param countValues The number of data elements to generate.
         * @param distr The random distribution to draw the data elements from.
         * @param sorted Whether the generated data shall be sorted.
         * @return An uncompressed column containing the generated data elements.
         * @todo Support also the random distributions returning real values, e.g.,
         * `std::normal_distribution`.
         */
        template< template< typename > class t_distr >
        static
        const column <uncompr_f> *
        generate_with_distr(size_t countValues, t_distr<uint64_t> distr, bool sorted, size_t seed = 0) {
            const size_t allocationSize = countValues * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(allocationSize);
            uint64_t * const res = resCol->get_data();
            if (seed == 0) {
                seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            }
            std::default_random_engine generator(seed);
            for (unsigned i = 0; i < countValues; i ++)
                res[i] = distr(generator);
        
            resCol->set_meta_data(countValues, allocationSize);
        
            if (sorted)
                std::sort(res, res + countValues);
        
            return resCol;
        }
    
    
        /**
         * @brief Creates an uncompressed column and fills its data buffer with values
         * generated according to the specified bit width histogram.
         *
         * Within each bit width, the values are drawn uniformly.
         *
         * @param p_CountValues The number of data elements to generate.
         * @param p_BwWeights An array of 64 64-bit numbers representing the bit width
         * histogram, or rather the weights of each bit width. The number of values of
         * bit width i to generate is the weight of bit width i multiplied by the the
         * number of data elements to generate.
         * @param p_IsSorted Whether the generated data shall be sorted.
         * @param p_IsExact
         * @param p_Seed The seed to use for the pseudo random number generator. If 0
         * (the default), then the current time will be used.
         * @return An uncompressed column containing the generated data elements.
         */
        static
        const column <uncompr_f> *
        generate_with_bitwidth_histogram(
          size_t p_CountValues,
          const uint64_t * p_BwWeights,
          bool p_IsSorted,
          bool p_IsExact,
          size_t p_Seed = 0
        ) {
            const size_t allocationSize = p_CountValues * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(allocationSize);
            uint64_t * res = resCol->get_data();
        
            if (p_Seed == 0) {
                p_Seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            }
            std::default_random_engine gen(p_Seed);
            std::uniform_int_distribution<uint64_t> distrVal(
              0, std::numeric_limits<uint64_t>::max()
            );
        
            const size_t digits = std::numeric_limits<uint64_t>::digits;
            size_t bwHist[digits];
        
            // Convert the given bit width weights to bit width histograms (weights to
            // absolute frequencies).
            uint64_t sumBwWeights = 0;
            for (unsigned bwMinus1 = 0; bwMinus1 < digits; bwMinus1 ++)
                sumBwWeights += p_BwWeights[bwMinus1];
            for (unsigned bwMinus1 = 0; bwMinus1 < digits; bwMinus1 ++)
                bwHist[bwMinus1] = static_cast<size_t>(round(
                  static_cast<double>(p_CountValues) * p_BwWeights[bwMinus1] / sumBwWeights
                ));
        
            // Due to rounding errors, the sum over the bit width histogram could
            // differ from the total number of data elements to generate (it could
            // be lower or higher. We compensate for that by either increasing or
            // decreasing buckets of the histogram in round-robin fashion until the
            // sum over the bit width histogram is exactly what we want. We only
            // change buckets that are non-zero (in order not to introduce new
            // bit widths) and non-max (in order to prevent integer overflows,
            // although this is extremely unlikely).
            size_t sumBwHist = 0;
            for (unsigned i = 0; i < digits; i ++)
                sumBwHist += bwHist[i];
            int correction = (sumBwHist < p_CountValues) ? 1 : - 1;
            for (unsigned i = 0; sumBwHist != p_CountValues; i = (i + 1) % digits) {
                if (bwHist[i] && ~ bwHist[i]) {
                    bwHist[i] += correction;
                    sumBwHist += correction;
                }
            }
        
            if (p_IsSorted) {
                for (unsigned bwMinus1 = 0; bwMinus1 < digits; bwMinus1 ++) {
                    uint64_t * const bwStart = res;
                    for (size_t i = 0; i < bwHist[bwMinus1]; i ++)
                        *res ++ = generate_with_bitwidth_histogram_helpers::get_value(
                          distrVal(gen), bwMinus1
                        );
                    std::sort(bwStart, res);
                }
            } else {
                // Generates (bw - 1).
                std::discrete_distribution<unsigned> distrBw(bwHist, bwHist + digits);
            
                if (p_IsExact) {
                    unsigned bwMinus1;
                    for (size_t i = 0; i < p_CountValues; i ++) {
                        do {
                            bwMinus1 = distrBw(gen);
                        } while (! bwHist[bwMinus1]);
                        bwHist[bwMinus1] --;
                        *res ++ = generate_with_bitwidth_histogram_helpers::get_value(
                          distrVal(gen), bwMinus1
                        );
                    }
                } else {
                    for (size_t i = 0; i < p_CountValues; i ++)
                        *res ++ = generate_with_bitwidth_histogram_helpers::get_value(
                          distrVal(gen), distrBw(gen)
                        );
                }
            }
        
            resCol->set_meta_data(p_CountValues, allocationSize);
        
            return resCol;
        }
    
    
        /**
         * @brief Creates an uncompressed column and fills its data buffer such that
         * exactly the specified number of data elements have the specified value.
         *
         * @param p_CountValues The number of data elements to generate.
         * @param p_CountMatches The exact number of occurences of the value
         * `p_ValMatch`.
         * @param p_ValMatch The value that shall appear exactly `p_CountMatches` times
         * in the data.
         * @param p_ValOther The value to use for all remaining data elements.
         * @param p_Seed The seed to use for the pseudo random number generator.
         * @return An uncompressed column containing the generated data elements.
         */
        static
        const column <uncompr_f> *
        generate_exact_number(
          size_t p_CountValues,
          size_t p_CountMatches,
          uint64_t p_ValMatch,
          uint64_t p_ValOther,
          bool p_Sorted,
          size_t p_Seed = 0
        ) {
            if (p_CountMatches > p_CountValues)
                throw std::runtime_error(
                  "p_CountMatches must be less than p_CountValues"
                );
            if (p_ValMatch == p_ValOther)
                throw std::runtime_error(
                  "p_ValMatch and p_ValOther must be different"
                );
        
            const size_t allocationSize = p_CountValues * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(allocationSize);
            uint64_t * const res = resCol->get_data();
        
            if (p_Sorted) {
                if (p_ValMatch < p_ValOther) {
                    for (size_t i = 0; i < p_CountMatches; i ++)
                        res[i] = p_ValMatch;
                    for (size_t i = p_CountMatches; i < p_CountValues; i ++)
                        res[i] = p_ValOther;
                } else {
                    for (size_t i = 0; i < p_CountValues - p_CountMatches; i ++)
                        res[i] = p_ValOther;
                    for (size_t i = p_CountValues - p_CountMatches; i < p_CountValues; i ++)
                        res[i] = p_ValMatch;
                }
            } else {
                // If the relative frequency is above 50%, then swap things to be more
                // efficient.
                if (p_CountMatches > p_CountValues / 2) {
                    p_CountMatches = p_CountValues - p_CountMatches;
                    const uint64_t tmp = p_ValMatch;
                    p_ValMatch = p_ValOther;
                    p_ValOther = tmp;
                }
            
                for (size_t i = 0; i < p_CountValues; i ++)
                    res[i] = p_ValOther;
            
                if (p_Seed == 0)
                    p_Seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                std::default_random_engine generator(p_Seed);
                std::uniform_int_distribution<size_t> rnd(0, p_CountValues - 1);
            
                for (size_t i = 0; i < p_CountMatches; i ++)
                    while (true) {
                        const size_t pos = rnd(generator);
                        if (res[pos] != p_ValMatch) {
                            res[pos] = p_ValMatch;
                            break;
                        }
                    }
            }
        
            resCol->set_meta_data(p_CountValues, allocationSize);
        
            return resCol;
        }
    
        static
        const column <uncompr_f> *
        generate_with_outliers_and_selectivity(size_t p_CountValues, uint64_t p_MainMin, uint64_t p_MainMax, double p_SelectedShare, uint64_t p_OutlierMin, uint64_t p_OutlierMax, double p_OutlierShare, bool p_IsSorted, size_t p_Seed = 0) {
            const bool mainAndOutliers = p_OutlierShare > 0 && p_OutlierShare < 1;
            if (! (p_MainMin < p_MainMax))
                throw std::runtime_error(
                  "p_MainMin < p_MainMax must hold"
                );
            if (mainAndOutliers && p_OutlierMin > p_OutlierMax)
                throw std::runtime_error(
                  "p_OutlierMin <= p_OutlierMax must hold if "
                  "0 < p_OutlierShare < 1"
                );
            if (mainAndOutliers && p_MainMax >= p_OutlierMin)
                throw std::runtime_error(
                  "p_MainMax < p_OutlierMin must hold if"
                  "0 < p_OutlierShare < 1"
                );
            if (p_SelectedShare < 0 || p_SelectedShare > 1)
                throw std::runtime_error(
                  "0 <= p_SelectedShare <= 1 must hold"
                );
            if (p_OutlierShare < 0 || p_OutlierShare > 1)
                throw std::runtime_error(
                  "0 <= p_OutlierShare <= 1 must hold"
                );
            if (p_SelectedShare + p_OutlierShare > 1)
                throw std::runtime_error(
                  "p_SelectedShare + p_OutlierShare <= 1 must hold"
                );
        
            const size_t allocationSize = p_CountValues * sizeof(uint64_t);
            auto resCol = new column<uncompr_f>(allocationSize);
            uint64_t * const res = resCol->get_data();
        
            if (p_Seed == 0)
                p_Seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::default_random_engine generator(p_Seed);
        
            std::bernoulli_distribution rndIsOutlier(p_OutlierShare);
            std::bernoulli_distribution rndIsSelected(
              p_SelectedShare / (1 - p_OutlierShare)
            );
            std::uniform_int_distribution<uint64_t> rndMain(p_MainMin + 1, p_MainMax);
            std::uniform_int_distribution<uint64_t> rndOutliers(
              p_OutlierMin, p_OutlierMax
            );
        
            for (size_t i = 0; i < p_CountValues; i ++) {
                if (rndIsOutlier(generator))
                    res[i] = rndOutliers(generator);
                else if (rndIsSelected(generator))
                    res[i] = p_MainMin;
                else
                    res[i] = rndMain(generator);
            }
        
            resCol->set_meta_data(p_CountValues, allocationSize);
        
            if (p_IsSorted)
                std::sort(res, res + p_CountValues);
        
            return resCol;
        }
    
    }; /// ColumnGenerator
/**
 * @brief Random number distribution that produces two different values.
 * 
 * The interface follows that of the distributions in the STL's `<random>`
 * header to the extend required for our data generation facilities.
 */
template<typename t_int_t>
class two_value_distribution {
    const t_int_t m_Val0;
    const t_int_t m_Val1;
    std::bernoulli_distribution m_Chooser;
    
    public:
        two_value_distribution(
                t_int_t p_Val0,
                t_int_t p_Val1,
                double p_ProbVal1
        ) :
                m_Val0(p_Val0),
                m_Val1(p_Val1)
        {
            m_Chooser = std::bernoulli_distribution(p_ProbVal1);
        }
        
        template<class t_generator_t>
        t_int_t operator()(t_generator_t & p_Generator) {
            return m_Chooser(p_Generator) ? m_Val1 : m_Val0;
        }
};

template<template<typename> class t_distr0_t, template<typename> class t_distr1_t>
struct two_distr_distribution {
    template<typename t_int_t>
    class distr {
        t_distr0_t<t_int_t> m_Distr0;
        t_distr1_t<t_int_t> m_Distr1;
        std::bernoulli_distribution m_Chooser;
        
    public:
        distr(
                t_distr0_t<t_int_t> p_Distr0,
                t_distr1_t<t_int_t> p_Distr1,
                double p_ProbVal1
        ) :
                m_Distr0(p_Distr0),
                m_Distr1(p_Distr1),
                m_Chooser(std::bernoulli_distribution(p_ProbVal1))
        {
            //
        }
        
        template<class t_generator_t>
        t_int_t operator()(t_generator_t & p_Generator) {
            return m_Chooser(p_Generator) ? m_Distr1(p_Generator) : m_Distr0(p_Generator);
        }
    };
};

/**
 * @brief A wrapper converting the values returned by non-integer distributions
 * to integers.
 * 
 * The parameters of the wrapped distribution should be chosen with care.
 * Generated values less than zero or greater than the maximum value of
 * `t_int_t` will be clipped.
 * 
 * Meant to be used with, e.g. std::normal_distribution or other distributions
 * returning floating point values.
 */
template<class t_distr_t>
struct int_distribution {
    // Since our data generation function generate_with_distr requires a
    // distribution class with exactly one template parameter, we need this
    // inner class.
    template<typename t_int_t>
    class distr {
        t_distr_t m_Distr;
        
    public:
        distr(t_distr_t p_Distr) : m_Distr(p_Distr) {
            //
        };

        template<class t_generator_t>
        MSV_CXX_ATTRIBUTE_INLINE
        t_int_t operator()(t_generator_t & p_Generator) {
#if 0
            // Variant 1) Clip values outside the range of the integer type to
            // zero and the maximum integer value, respectively. Sometimes
            // results in unexpected values less than zero.
            const typename t_distr_t::result_type res =
                    std::round(m_Distr(p_Generator));
            if(res < 0)
                return 0;
            const t_int_t max = std::numeric_limits<t_int_t>::max();
            if(res > max)
                return max;
#else
            // Variant 2) Discard values outside the range of the integer type.
            // Might result in an infinite loop or a very slow data generation
            // if the normal distribution has unsuitable parameters.
            const t_int_t max = std::numeric_limits<t_int_t>::max();
            typename t_distr_t::result_type res;
            do {
                res = std::round(m_Distr(p_Generator));
            } while(res < 0 || res > max);
#endif
            return static_cast<t_int_t>(res);
        }
    };
};




}
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_GEN_H
