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
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace morphstore {
    
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
const column<uncompr_f> * make_column(const std::vector<uint64_t> & vec) {
    const size_t count = vec.size();
    if(count > 20)
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

const column<uncompr_f> * make_column(uint64_t const * const vec, size_t count) {
   if(count > 400)
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
const column<uncompr_f> * generate_sorted_unique(
        size_t countValues,
        uint64_t start = 0,
        uint64_t step = 1
) {
    const size_t allocationSize = countValues * sizeof(uint64_t);
    auto resCol = new column<uncompr_f>(allocationSize);
    uint64_t * const res = resCol->get_data();
    
    for(unsigned i = 0; i < countValues; i++)
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
const column<uncompr_f> * generate_sorted_unique_extraction(
        size_t p_CountValues,
        size_t p_CountPopulation,
        size_t p_Seed = 0
) {
    const size_t allocationSize = p_CountValues * sizeof(uint64_t);
    auto resCol = new column<uncompr_f>(allocationSize);
    uint64_t * res = resCol->get_data();
    
    if(p_Seed == 0)
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
    if(!flip) { // Select values.
        size_t countChosen = 0;
        while(countChosen < p_CountValues) {
            const uint64_t val = distr(gen);
            if(!chosen[val]) {
                chosen[val] = true;
                countChosen++;
            }
        }
    }
    else { // Unselect values.
        size_t countUnChosen = 0;
        while(countUnChosen < p_CountPopulation - p_CountValues) {
            const uint64_t val = distr(gen);
            if(chosen[val]) {
                chosen[val] = false;
                countUnChosen++;
            }
        }
    }
    
    for(size_t i = 0; i < p_CountPopulation; i++)
        if(chosen[i])
            *res++ = i;
    
    resCol->set_meta_data(p_CountValues, allocationSize);
    
    return resCol;
}

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
template<template<typename> class t_distr>
const column<uncompr_f> * generate_with_distr(
        size_t countValues,
        t_distr<uint64_t> distr,
        bool sorted,
        size_t seed = 0
) {
    const size_t allocationSize = countValues * sizeof(uint64_t);
    auto resCol = new column<uncompr_f>(allocationSize);
    uint64_t * const res = resCol->get_data();
    if( seed == 0 ) {
       seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    std::default_random_engine generator(
         seed
    );
    for(unsigned i = 0; i < countValues; i++)
        res[i] = distr(generator);
    
    resCol->set_meta_data(countValues, allocationSize);
    
    if(sorted)
        std::sort(res, res + countValues);
    
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
const column<uncompr_f> * generate_exact_number(
        size_t p_CountValues,
        size_t p_CountMatches,
        uint64_t p_ValMatch,
        uint64_t p_ValOther,
        size_t p_Seed = 0
) {
    if(p_CountMatches > p_CountValues)
        throw std::runtime_error(
                "p_CountMatches must be less than p_CountValues"
        );
    if(p_ValMatch == p_ValOther)
        throw std::runtime_error(
                "p_ValMatch and p_ValOther must be different"
        );
    
    // If the relative frequency is above 50%, then swap things to be more
    // efficient.
    if(p_CountMatches > p_CountValues / 2)
        return generate_exact_number(
                p_CountValues,
                p_CountValues - p_CountMatches,
                p_ValOther,
                p_ValMatch,
                p_Seed
        );
    
    const size_t allocationSize = p_CountValues * sizeof(uint64_t);
    auto resCol = new column<uncompr_f>(allocationSize);
    uint64_t * const res = resCol->get_data();
    
    for(size_t i = 0; i < p_CountValues; i++)
        res[i] = p_ValOther;
    
    if(p_Seed == 0)
       p_Seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(p_Seed);
    std::uniform_int_distribution<size_t> rnd(0, p_CountValues - 1);
    
    for(size_t i = 0; i < p_CountMatches; i++)
        while(true) {
            const size_t pos = rnd(generator);
            if(res[pos] != p_ValMatch) {
                res[pos] = p_ValMatch;
                break;
            }
        }
    
    resCol->set_meta_data(p_CountValues, allocationSize);
    
    return resCol;
}

const column<uncompr_f> * generate_with_outliers_and_selectivity(
        size_t p_CountValues,
        uint64_t p_MainMin, uint64_t p_MainMax,
        double p_SelectedShare,
        uint64_t p_OutlierMin, uint64_t p_OutlierMax, double p_OutlierShare,
        bool p_IsSorted,
        size_t p_Seed = 0
) {
    if(!(p_MainMin < p_MainMax))
        throw std::runtime_error(
                "p_MainMin < p_MainMax must hold"
        );
    if(p_OutlierShare && p_OutlierMin > p_OutlierMax)
        throw std::runtime_error(
                "p_OutlierMin <= p_OutlierMax must hold if p_OutlierShare > 0"
        );
    if(p_OutlierShare && p_MainMax >= p_OutlierMin)
        throw std::runtime_error(
                "p_MainMax < p_OutlierMin must hold if p_OutlierShare > 0"
        );
    if(p_SelectedShare < 0 || p_SelectedShare > 1)
        throw std::runtime_error(
                "0 <= p_SelectedShare <= 1 must hold"
        );
    if(p_OutlierShare < 0 || p_OutlierShare > 1)
        throw std::runtime_error(
                "0 <= p_OutlierShare <= 1 must hold"
        );
    if(p_SelectedShare + p_OutlierShare > 1)
        throw std::runtime_error(
                "p_SelectedShare + p_OutlierShare <= 1 must hold"
        );
    
    const size_t allocationSize = p_CountValues * sizeof(uint64_t);
    auto resCol = new column<uncompr_f>(allocationSize);
    uint64_t * const res = resCol->get_data();
    
    if(p_Seed == 0)
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
    
    for(size_t i = 0; i < p_CountValues; i++) {
        if(rndIsOutlier(generator))
            res[i] = rndOutliers(generator);
        else if(rndIsSelected(generator))
            res[i] = p_MainMin;
        else
            res[i] = rndMain(generator);
    }
    
    resCol->set_meta_data(p_CountValues, allocationSize);
    
    if(p_IsSorted)
        std::sort(res, res + p_CountValues);
    
    return resCol;
}

}
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_GEN_H
