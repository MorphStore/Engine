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
 * @file variant_executor.h
 * @brief 
 * @todo Documentation.
 * @todo Support for repetitions.
 * @todo Rethink where the cached columns should be freed.
 * @todo Include data generation, so that it can be included in the outputs.
 * Then we would not need printDataGenStarted() and printDataGenDone() anymore.
 * @todo Print the monitoring (CSV) output after each call to execute_variants
 * to make experiments abortable without losing all data measured so far.
 * @todo It would be cool if even segfaults in a variant could be tolerated.
 */

#ifndef MORPHSTORE_CORE_UTILS_VARIANT_EXECUTOR_H
#define MORPHSTORE_CORE_UTILS_VARIANT_EXECUTOR_H

#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/column_cache.h>
#include <core/utils/equality_check.h>
#include <core/utils/monitoring.h>
#include <core/utils/preprocessor.h>
#include <core/utils/printing.h>
#include <core/utils/processing_style.h>
#include <core/utils/variadic.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace morphstore {
    
    class variant_executor {
        static const std::string m_CsvColNameRuntime;
        static const std::string m_CsvColNameCheck;
        
    public:
        // These must all be uncompr_f, but their number matters.
        template<class ... t_uncompr_out_fs>
        class for_uncompr_output_formats {
    
            template<size_t ... t_Idxs>
            static bool check_uncompr_column_tuples(
                    std::tuple<const column<t_uncompr_out_fs> * ...> p_ExpCols,
                    std::tuple<const column<t_uncompr_out_fs> * ...> p_FndCols,
                    MSV_CXX_ATTRIBUTE_PPUNUSED std::index_sequence<t_Idxs ...>
            ) {
                STATIC_ASSERT_PARAMPACK_SAMESIZE(t_Idxs, t_uncompr_out_fs)
                std::vector<bool> vec = {
                    equality_check(
                            std::get<t_Idxs>(p_ExpCols),
                            std::get<t_Idxs>(p_FndCols)
                    ).good() ...
                };
                return std::count(vec.begin(), vec.end(), false) == 0;
            }
            
            template<size_t ... t_Idxs>
            static void delete_uncompr_column_tuple(
                    std::tuple<const column<t_uncompr_out_fs> * ...> p_Cols,
                    MSV_CXX_ATTRIBUTE_PPUNUSED std::index_sequence<t_Idxs ...>
            ) {
                STATIC_ASSERT_PARAMPACK_SAMESIZE(t_Idxs, t_uncompr_out_fs)
                std::vector<const column<uncompr_f> *> vec = {
                    std::get<t_Idxs>(p_Cols) ...
                };
                for(auto col : vec)
                    delete col;
            }
            
        public:
            // These must all be uncompr_f, but their number matters.
            template<class ... t_uncompr_in_fs>
            struct for_uncompr_input_formats {
                
                template<typename ... t_additional_param_ts>
                struct for_additional_params {

                    template<typename ... t_variant_param_ts>
                    struct for_variant_params {

                        template<typename ... t_setting_param_ts>
                        class for_setting_params {

                            const std::vector<std::string> m_CsvVariantParamColNames;
                            const std::vector<std::string> m_CsvSettingParamColNames;
                            std::vector<std::string> m_CsvAllParamColNames;
                            bool m_AllGood;

                            void check_csv_column_names(
                                    const std::vector<std::string> & p_CsvColNames,
                                    const std::string & p_Word,
                                    size_t p_ExpectedCount
                            ) {
                                const size_t count = p_CsvColNames.size();
                                if(count != p_ExpectedCount) {
                                    std::stringstream s;
                                    s
                                            << "you must provide as many " << p_Word
                                            << " column names as there are " << p_Word
                                            << "s (found " << count << ", expected "
                                            << p_ExpectedCount << ')';
                                    throw std::runtime_error(s.str());
                                }
                            }

                            // We need some return value so that we can use
                            // this function in a parameter pack expansion.
                            template<typename t_type>
                            bool print_setting_param(unsigned & p_SettingParamIdx, t_type p_Val) {
                                std::cerr
                                        << "\t\t" << m_CsvSettingParamColNames[p_SettingParamIdx++]
                                        << ": \t" << p_Val << std::endl;
                                return false;
                            }

                        public:
                            struct abstract_operator_wrapper {
                                virtual
                                std::tuple<const column<t_uncompr_out_fs> * ...>
                                call_operator(
                                        column_cache & cache,
                                        const column<t_uncompr_in_fs> * ... p_InCols,
                                        t_additional_param_ts ... p_AdditionalParams,
                                        t_variant_param_ts ... p_VariantParams,
                                        t_setting_param_ts ... p_SettingParams
                                ) = 0;
                            };

                            struct operator_wrapper {

                                // return_type_helper is required, because those
                                // operators that return a single column return it as
                                // a mere pointer, not as a tuple containing a pointer.

                                template<class ... t_out_fs>
                                struct return_type_helper {
                                    STATIC_ASSERT_PARAMPACK_SAMESIZE(t_out_fs, t_uncompr_out_fs)
                                    using return_t = std::tuple<const column<t_out_fs> * ...>;
                                };

                                template<class t_single_out_f>
                                struct return_type_helper<t_single_out_f> {
                                    using return_t = const column<t_single_out_f> *;
                                };

                                template<class ... t_out_fs>
                                struct for_output_formats {

                                    STATIC_ASSERT_PARAMPACK_SAMESIZE(t_out_fs, t_uncompr_out_fs)

                                    template<size_t ... t_Idxs>
                                    static
                                    std::tuple<const column<t_uncompr_out_fs> * ...>
                                    decompress_column_tuple(
                                            std::tuple<const column<t_out_fs> * ...> p_Cols,
                                            MSV_CXX_ATTRIBUTE_PPUNUSED std::index_sequence<t_Idxs ...>
                                    ) {
                                        STATIC_ASSERT_PARAMPACK_SAMESIZE(t_Idxs, t_uncompr_out_fs)
                                        return {
                                            // @todo Do not hardcode the processing style.
                                            morph<processing_style_t::scalar, uncompr_f>(
                                                    std::get<t_Idxs>(p_Cols) ...
                                            )
                                        };
                                    }

                                    template<size_t t_Count, class t_head_f, class ... t_tail_fs>
                                    struct column_tuple_deleter {
                                        static void apply(
                                                const column<t_head_f> * p_HeadCol,
                                                const column<t_tail_fs> * ... p_TailCols
                                        ) {
                                            if(std::is_same<t_head_f, uncompr_f>::value)
                                                delete p_HeadCol;
                                            column_tuple_deleter<t_Count - 1, t_tail_fs ...>::apply(
                                                    p_TailCols ...
                                            );
                                        }
                                    };

                                    template<class t_head_f, class ... t_tail_fs>
                                    struct column_tuple_deleter<1, t_head_f, t_tail_fs ...> {
                                        static void apply(
                                                const column<t_head_f> * p_HeadCol,
                                                const column<t_tail_fs> * ... p_TailCols
                                        ) {
                                            if(typeid(t_head_f) != typeid(uncompr_f))
                                                delete p_HeadCol;
                                        }
                                    };

                                    template<size_t ... t_Idxs>
                                    static void delete_column_tuple(
                                            std::tuple<const column<t_out_fs> * ...> p_Cols,
                                            MSV_CXX_ATTRIBUTE_PPUNUSED std::index_sequence<t_Idxs ...>
                                    ) {
                                        STATIC_ASSERT_PARAMPACK_SAMESIZE(t_Idxs, t_out_fs)
                                        column_tuple_deleter<sizeof...(t_out_fs), t_out_fs ...>::apply(
                                                std::get<t_Idxs>(p_Cols) ...
                                        );
                                    }

                                    template<class ... t_in_fs>
                                    struct for_input_formats : public abstract_operator_wrapper {

                                        STATIC_ASSERT_PARAMPACK_SAMESIZE(t_in_fs, t_uncompr_in_fs)

                                        using op_func_ptr_t = typename return_type_helper<t_out_fs ...>::return_t (*)(
                                                const column<t_in_fs> * ...,
                                                t_additional_param_ts ...
                                        );

                                        op_func_ptr_t m_OpFuncPtr;

                                        for_input_formats(op_func_ptr_t p_OpFuncPtr)
                                        : m_OpFuncPtr(p_OpFuncPtr) {
                                            //
                                        }

                                        std::tuple<const column<t_uncompr_out_fs> * ...>
                                        call_operator(
                                                column_cache & cache,
                                                const column<t_uncompr_in_fs> * ... p_InCols,
                                                t_additional_param_ts ... p_AdditionalParams,
                                                MSV_CXX_ATTRIBUTE_PPUNUSED t_variant_param_ts ... p_VariantParams,
                                                MSV_CXX_ATTRIBUTE_PPUNUSED t_setting_param_ts ... p_SettingParams
                                        ) {
                                            std::make_tuple(
                                                    cache.ensure_presence<t_in_fs>(p_InCols) ...
                                            );
                                            MONITORING_START_INTERVAL_FOR(
                                                    m_CsvColNameRuntime,
                                                    p_VariantParams ...,
                                                    p_SettingParams ...,
                                                    p_AdditionalParams ...
                                            );
                                            auto resInternal = (*m_OpFuncPtr)(
                                                    cache.get<t_in_fs>(p_InCols) ...,
                                                    p_AdditionalParams ...
                                            );
                                            MONITORING_END_INTERVAL_FOR(
                                                    m_CsvColNameRuntime,
                                                    p_VariantParams ...,
                                                    p_SettingParams ...,
                                                    p_AdditionalParams ...
                                            );
                                            // Note that, when `m_OpFuncPtr` returns a
                                            // single pointer (not wrapped in a tuple),
                                            // the following lines are totally ok,
                                            // although the first parameter of
                                            // `decompress_column_tuple` is a tuple,
                                            // since this single-element-tuple can be
                                            // constructed from a single pointer passed
                                            // to it. It is essentially the same thing
                                            // as `std::tuple<int> x = 1;`, which is
                                            // also valid.
                                            auto seq = std::index_sequence_for<t_out_fs ...>();
                                            auto resDecompr = decompress_column_tuple(resInternal, seq);
                                            delete_column_tuple(resInternal, seq);
                                            return resDecompr;
                                        }
                                    };
                                };
                            };

                            using variant_t = std::tuple<
                                    abstract_operator_wrapper *,
                                    t_variant_param_ts ...
                            >;
                            
                            using setting_t = std::tuple<t_setting_param_ts ...>;

                            for_setting_params (
                                    const std::vector<std::string> p_AddParamsCsvParamColNames,
                                    const std::vector<std::string> p_VariantCsvParamColNames,
                                    const std::vector<std::string> p_SettingCsvParamColNames
                            ) :
                                    m_CsvVariantParamColNames(p_VariantCsvParamColNames),
                                    m_CsvSettingParamColNames(p_SettingCsvParamColNames),
                                    m_AllGood(true)
                            {
                                check_csv_column_names(
                                        p_VariantCsvParamColNames,
                                        "variant parameter",
                                        sizeof...(t_variant_param_ts)
                                );
                                check_csv_column_names(
                                        p_SettingCsvParamColNames,
                                        "setting parameter",
                                        sizeof...(t_setting_param_ts)
                                );
                                check_csv_column_names(
                                        p_AddParamsCsvParamColNames,
                                        "additional parameter",
                                        sizeof...(t_additional_param_ts)
                                );

                                m_CsvAllParamColNames = p_VariantCsvParamColNames;
                                m_CsvAllParamColNames.insert(
                                        m_CsvAllParamColNames.end(),
                                        p_SettingCsvParamColNames.begin(),
                                        p_SettingCsvParamColNames.end()
                                );
                                m_CsvAllParamColNames.insert(
                                        m_CsvAllParamColNames.end(),
                                        p_AddParamsCsvParamColNames.begin(),
                                        p_AddParamsCsvParamColNames.end()
                                );
                            }

                        private:
                            template<size_t ... t_VariantParamIdxs>
                            void execute_variants_internal(
                                    const std::vector<variant_t> p_Variants,
                                    t_setting_param_ts ... p_SettingParams,
                                    const column<t_uncompr_in_fs> * ... p_InCols,
                                    t_additional_param_ts ... p_AdditionalParams,
                                    MSV_CXX_ATTRIBUTE_PPUNUSED std::index_sequence<t_VariantParamIdxs ...>
                            ) {
                                STATIC_ASSERT_PARAMPACK_SAMESIZE(t_VariantParamIdxs, t_variant_param_ts)

                                column_cache cache;

                                std::cerr
                                        << "Setting" << std::endl
                                        << "\tParameters" << std::endl;
                                {
                                    if(sizeof...(t_setting_param_ts)) {
                                        MSV_CXX_ATTRIBUTE_PPUNUSED unsigned i = 0;
                                        MSV_CXX_ATTRIBUTE_PPUNUSED bool x[] = {
                                            // false is needed, because p_SettingParams could be empty.
                                            false, print_setting_param(i, p_SettingParams) ...
                                        };
                                    }
                                    else
                                        std::cerr << "\t\t(none)" << std::endl;
                                }
                                
                                // Calculate the maximum width of each column.
                                size_t maxVariantParamWs[] = {get_text_length(m_CsvVariantParamColNames[t_VariantParamIdxs]) ...};
                                for(auto variant : p_Variants) {
                                    size_t variantParamWs[] = {get_text_length(std::get<t_VariantParamIdxs + 1>(variant)) ...};
                                    for(unsigned vpIdx = 0; vpIdx < sizeof...(t_VariantParamIdxs); vpIdx++)
                                        if(variantParamWs[vpIdx] > maxVariantParamWs[vpIdx])
                                            maxVariantParamWs[vpIdx] = variantParamWs[vpIdx];
                                }
                                const std::string vpDelim = "  ";

                                std::cerr
                                        << "\tExecuting Variants" << std::endl
                                        << "\t\t";
                                for(unsigned vpIdx = 0; vpIdx < sizeof...(t_VariantParamIdxs); vpIdx++)
                                    std::cerr
                                            << std::left << std::setw(maxVariantParamWs[vpIdx])
                                            << m_CsvVariantParamColNames[vpIdx] << vpDelim;
                                std::cerr << std::endl;
                                auto nullptrTuple = repeat_as_tuple<
                                        sizeof...(t_uncompr_out_fs),
                                        const column<uncompr_f> *,
                                        nullptr
                                >::value;
                                std::tuple<const column<t_uncompr_out_fs> * ...> referenceOutput = nullptrTuple;
                                bool allGood = true;
                                for(auto variant : p_Variants) {
                                    abstract_operator_wrapper * op = std::get<0>(variant);

                                    std::cerr
                                            << "\t\t"
                                            << doPrintWithWidths(vpDelim, maxVariantParamWs, std::get<t_VariantParamIdxs + 1>(variant) ...)
                                            << ": started... ";
                                    std::cerr.flush();

                                    MONITORING_CREATE_MONITOR(
                                            MONITORING_MAKE_MONITOR(
                                                    std::get<t_VariantParamIdxs + 1>(variant) ...,
                                                    p_SettingParams ...,
                                                    p_AdditionalParams ...
                                            ),
                                            m_CsvAllParamColNames
                                    );

                                    auto currentOutput = op->call_operator(
                                            cache,
                                            p_InCols ...,
                                            p_AdditionalParams ...,
                                            std::get<t_VariantParamIdxs + 1>(variant) ...,
                                            p_SettingParams ...
                                    );
                                    // @todo At the moment, we have to clear the column cache after the execution
                                    // of each operator variant in order not to use too much memeory when there
                                    // are too many variants. We should selectively free certain formats when they
                                    // are not needed any more.
                                    cache.clear();

                                    std::cerr << "done.";

                                    if(referenceOutput == nullptrTuple) {
                                        referenceOutput = currentOutput;
                                        MONITORING_ADD_INT_FOR(
                                                m_CsvColNameCheck,
                                                -1,
                                                std::get<t_VariantParamIdxs + 1>(variant) ...,
                                                p_SettingParams ...,
                                                p_AdditionalParams ...
                                        );
                                        std::cerr << " -> " << colored("reference", colored::color::cyan);
                                    }
                                    else {
                                        const bool good = check_uncompr_column_tuples(
                                                referenceOutput,
                                                currentOutput,
                                                std::index_sequence_for<t_uncompr_out_fs ...>()
                                        );
                                        allGood = allGood && good;
                                        MONITORING_ADD_INT_FOR(
                                                m_CsvColNameCheck,
                                                good,
                                                std::get<t_VariantParamIdxs + 1>(variant) ...,
                                                p_SettingParams ...,
                                                p_AdditionalParams ...
                                        );
                                        // @todo One of the output columns could be one of the input columns
                                        // and then it must not be freed (otherwise segfault) (this can
                                        // happen, when a no-op morph-operator is among the variants).
                                        delete_uncompr_column_tuple(
                                                currentOutput,
                                                std::index_sequence_for<t_uncompr_out_fs ...>()
                                        );
                                        std::cerr << " -> " << (
                                                good
                                                ? colored("ok", colored::color::green)
                                                : colored("not ok", colored::color::red)
                                        );
                                    }
                                    std::cerr << std::endl;
                                }
                                
                                // @todo One of the output columns could be one of the input columns
                                // and then it must not be freed (otherwise segfault) (this can
                                // happen, when a no-op morph-operator is among the variants).
                                delete_uncompr_column_tuple(
                                        referenceOutput,
                                        std::index_sequence_for<t_uncompr_out_fs ...>()
                                );
                                m_AllGood = m_AllGood && allGood;
                            }

                        public:
                            void execute_variants(
                                    const std::vector<variant_t> p_Variants,
                                    t_setting_param_ts ... p_SettingParams,
                                    const column<t_uncompr_in_fs> * ... p_InCols,
                                    t_additional_param_ts ... p_AdditionalParams
                            ) {
                                execute_variants_internal(
                                        p_Variants,
                                        p_SettingParams ...,
                                        p_InCols ...,
                                        p_AdditionalParams ...,
                                        std::index_sequence_for<t_variant_param_ts ...>()
                                );
                            }

                            void done() const {
                                std::cerr
                                        << "Summary" << std::endl
                                        << '\t' << (
                                                m_AllGood
                                                ? colored("all ok", colored::color::green)
                                                : colored("SOME NOT OK", colored::color::red)
                                        )
                                        << std::endl << std::endl;
                                MONITORING_PRINT_MONITORS(monitorCsvLog);
                            }

                            bool good() const {
                                return m_AllGood;
                            }

                            void print_datagen_started() const {
                                std::cerr << "Data generation: started... ";
                            };

                            void print_datagen_done() const {
                                std::cerr << "done." << std::endl;
                            }
                        };
                    };
                };
            };
        };
    };

    const std::string variant_executor::m_CsvColNameRuntime = "runtime:µs";
    const std::string variant_executor::m_CsvColNameCheck = "check";
    
    template<
            unsigned t_OutputColumnCount,
            unsigned t_InputColumnCount,
            typename ... t_additional_param_ts
    >
    class variant_executor_helper {
        
        template<template<class ...> class t_target, unsigned t_Count, class ... t_uncompr_fs>
        struct repeat_uncompr_f : public repeat_uncompr_f<t_target, t_Count - 1, uncompr_f, t_uncompr_fs ...> {
            //
        };
        template<template<class...> class t_target, class ... t_uncompr_fs>
        struct repeat_uncompr_f<t_target, 0, t_uncompr_fs ...> {
            using type = t_target<t_uncompr_fs ...>;
        };
        
    public:
        using type = typename repeat_uncompr_f<
                repeat_uncompr_f<
                        variant_executor::for_uncompr_output_formats,
                        t_OutputColumnCount
                >::type::template for_uncompr_input_formats,
                t_InputColumnCount
        >::type::template for_additional_params<t_additional_param_ts ...>;
    };
}
#endif //MORPHSTORE_CORE_UTILS_VARIANT_EXECUTOR_H
