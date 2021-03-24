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


#ifndef MORPHSTORE_INCLUDE_CORE_VIRTUAL_METAOPERATOR_H
#define MORPHSTORE_INCLUDE_CORE_VIRTUAL_METAOPERATOR_H

#include <interfaces>
#include <utils>
#include <vectorExtensions>
#include <core/virtual/Executable.h>

namespace morphstore {
    using namespace virtuallib;
    using namespace vectorlib;
    
    
    
    
    template<IVectorExtension TVectorExtension, IPartitioner TPartitioner, IExecutable TExecutable, IPartitioner TConsolidator>
    class MetaOperator : public Executable {
      
      public:
        /// @todo default implementation for non-virtual vector extensions
    };
    
    template<IVectorBuilder TVectorBuilder, IPartitioner TPartitioner, IExecutable TExecutable, IPartitioner TConsolidator>
    class MetaOperator<vv<TVectorBuilder>, TPartitioner, TExecutable, TConsolidator> : public Executable {
      public:
        /// @todo TOFIX: results in column* instead of PartitionedColumn*, which is needed for Executor<...> or reduce Executor template parameters
        using executableInputList_t = typename unfold_type(TExecutable::apply)::inputType;
        
        using executableOutput_t = typename pack<typename unfold_type(TExecutable::apply)::returnType>::type;
//        using executor = Executor<TVectorBuilder::ctype, TExecutable, executableOutputTypes, executableInputTypes>;

        using executor_t = Executor<TVectorBuilder::ctype, TExecutable>;
        using output_t = PartitionedColumn<TConsolidator, uncompr_f>*;

//        static
//        void apply(executableInputTypes input){
////            TExecutable::apply(input);
//        }
    
    
      private:
        template<IStorage TStorage>
        static auto prepare(TStorage * storage){
            /// create a PartitionedColumn from given storage (e.g. column<...>)
            return new PartitionedColumn<
                TPartitioner,
                typename TStorage::format
            >(storage, TVectorBuilder::cvalue);
        }
        
        
        /**
         * Pass through arithmetics.
         * @tparam TValue
         * @param value
         * @return
         */
        template<IArithmetic TValue>
        static auto prepare(TValue value){
            return value;
        }
        
        /**
         * Pass through arithmetic pointers.
         * @tparam TValue
         * @param value
         * @return
         */
        template<IArithmetic TValue>
        static auto prepare(TValue * value){
            return value;
        }
      
      public:
        template<typename...TInputList>
        requires std::is_same<std::tuple<TInputList...>, executableInputList_t>::value
        static
        PartitionedColumn<TConsolidator, uncompr_f>* apply(TInputList...inputList){
            /// @todo: create PartitionedColumn from each input .... values?
            
//            static_assert(
//              std::is_same<std::tuple<TInputList...>, inputListExecutable>::value,
//              "Error: Arguments of MetaOperator::apply do not satisfy"
//            );
            
//            executor::template apply(uint64_t(3), nullptr, uint64_t(5));

//            std::cout << "Before prepare: " << (type_str<decltype(inputList)...>::apply()) << std::endl;
//            std::cout << "After prepare: " << (type_str<decltype(prepare(inputList))...>::apply()) << std::endl;
//            std::cout << "Executor interface: " << (typestr(executor::template apply<column<uncompr_f>*, uint64_t>)) << std::endl;
//            std::cout << "Prepare function: " << (typestr(prepare<uint64_t>)) << std::endl;
            
//            std::cout << "Prepare::apply: " << typestr(Prepare<uint64_t>::apply) << std::endl;
            
//            std::cout << "Converted types: " << (typestr<
//              convert_each_type_using_function<decltype(MetaOperator<vv<TVectorBuilder>, TPartitioner, TExecutable, TConsolidator>::prepare), inputList...>::type
//              >())) << std::endl;

            /// @todo do things
            
            
//            std::cout << "Size of inputList: " << sizeof...(inputList) << std::endl;
            
            
//            std::cout << "Elements inputList: ";
//            ((std::cout << inputList << ", "), ...);
//            std::cout << " // ";
//            ((std::cout << typestr(inputList) << ", "), ...);
//            std::cout << std::endl;
            
            
//            std::cout << "Elements inputList after prepare: ";
//            ((std::cout << prepare(inputList) << ", "), ...);
//            std::cout << " // ";
//            ((std::cout << typestr(prepare(inputList)) << ", "), ...);
//            std::cout << std::endl;
            
            /// Run Executor
            std::vector< typename executor_t::output_t * > *
            result_executor = executor_t::apply(
              size_t(TVectorBuilder::cvalue), /// number of threads to spawn
              prepare(inputList) ... /// prepare each input, depending on its type
              );
            
            /// Store result in PartitionedColumn
            auto result = new PartitionedColumn<TConsolidator, uncompr_f>();
            for(auto& outputN : *result_executor){
                result->addPartition(*outputN);
            }
//            std::cout << typestr<executor>() << std::endl;
//            std::cout << demangle(typeid(executor).name()) << std::endl;
//            std::cout << typeid(executor).name() << std::endl;
            
            return result;
        }
    };
    
} // namespace
#endif //MORPHSTORE_INCLUDE_CORE_VIRTUAL_METAOPERATOR_H
