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


#ifndef MORPHSTORE_INCLUDE_CORE_VIRTUAL_EXECUTOR_H
#define MORPHSTORE_INCLUDE_CORE_VIRTUAL_EXECUTOR_H

#include <forward>
#include <stdlibs>
#include <utils>
#include <vectorExtensions>

namespace morphstore {
    
    using namespace virtuallib;
    #ifdef USE_CPP20_CONCEPTS
    template<typename Type>
    concept Is_Pointer = std::is_pointer<Type>::value;
    template<typename Type>
    concept Is_No_Pointer = !std::is_pointer<Type>::value;
    #else
      #define Is_Pointer typename
      #define Is_No_Pointer typename
    #endif
    
    
    /// getter for pointer and non-pointer types
    template<size_t index, Is_Pointer Type>
    decltype( ((Type) nullptr)->operator[](index) )
    get(Type t) {
        return (*t)[index];
    }
    template<size_t index, Is_No_Pointer Type>
    decltype( ((Type*) nullptr)->operator[](index) )
    get(Type t) {
        return t[index];
    }
    /// non-static variant of getters
    template<Is_Pointer Type>
    decltype( ((Type) nullptr)->operator[](size_t(0)) )
    get(Type t, size_t index){
        return (*t)[index];
    }
    template<Is_No_Pointer Type>
    decltype( ((Type*) nullptr)->operator[](size_t(0)) )
    get(Type t, size_t index){
        return t[index];
    }
    template<IArithmetic Type>
    Type
    get(Type& t, size_t index){
        return t;
    }
    template<IArithmeticPtr Type>
    Type
    get(Type t, size_t index){
        return t;
    }
    
    
    struct __print_type {
        template<typename T>
        static void apply(T t) {
            std::cout << "Unknown type. Typeid: " << typeid(T).name() << std::endl;
        }
        
        static void apply(uint64_t t){
            std::cout << "Type is uint64_t." << std::endl;
        }
        static void apply(uint64_t* t){
            std::cout << "Type is uint64_t*." << std::endl;
        }
        template<typename ... Ts>
        static void apply(PartitionedColumn<Ts...> t){
            std::cout << "Type is PartitionedColumn<...>." << std::endl;
        }
        template<typename ... Ts>
        static void apply(PartitionedColumn<Ts...>* t){
            std::cout << "Type is PartitionedColumn<...>*." << std::endl;
        }
        
        template<typename T>
        static void apply(column<T> t){
            std::cout << "Type is column<...>." << std::endl;
        }
        template<typename T>
        static void apply(column<T>* t){
            std::cout << "Type is column<...>*." << std::endl;
        }
    };

    template<typename T>
    uint64_t print_type(T t){
        __print_type::apply(t);
        return 0;
    }
    template<typename T, typename ...TOther>
    uint64_t print_type(T t, TOther...other){
        __print_type::apply(t);
        print_type(other...);
        return 0;
    }
    uint64_t print_type(){
        return 0;
    }
    
    

    template<typename ... TInputs>
    struct InputTypes : public std::tuple<TInputs...> {
        InputTypes(TInputs...inputs) : std::tuple<TInputs...>(inputs...) {};
    };
    template<typename ... TOutputs>
    struct OutputTypes : public std::tuple<TOutputs...> {
        OutputTypes(TOutputs...outputs) : std::tuple<TOutputs...>(outputs...) {};
    };
    
    
    
    
    
    template<IExecutable TExecutable, typename ... TInAndOutputTypes>
    class ExecutorCore {
        static_assert(always_false<TInAndOutputTypes...>::value, "Unsupported types");
    };
    
    template<IExecutable TExecutable, IOperatorInput ... TInputTypes, template<typename...> class TIn>
      requires std::is_same< TIn<TInputTypes...>, typename unfold_type(TExecutable::apply)::inputType >::value
    class ExecutorCore<TExecutable, TIn<TInputTypes...>> {
//        using output_t = OutputTypes<typename unfold_type(TExecutable::apply)::returnType>;
//        using output_t = typename pack<typename unfold_type(TExecutable::apply)::returnType>::type;
        using output_t = typename unfold_type(TExecutable::apply)::returnType;
        using input_t = typename unfold_type(TExecutable::apply)::inputType;
      public:
        
        static void apply(output_t ** output, TInputTypes ... inputs){
//            std::cout << "Run ExecutorCore::apply()" << std::endl;
            *output = new output_t(TExecutable::apply(inputs...));
        }
    };
    
    template<ConcurrentType Tctype, IExecutable TExecutable>
    class Executor : public Executable {
        static_assert(always_false<TExecutable>::value, "Unsupported concurrent type");
    };
    
    
    /// std::thread variant
    template< IExecutable TExecutable>
    class Executor<ConcurrentType::STD_THREADS, TExecutable> : public Executable {
      public:
//        using outType_t = OutputTypes<typename unfold_type(TExecutable::apply)::returnType>;
//        using output_t = typename pack<typename unfold_type(TExecutable::apply)::returnType>::type;
        using output_t = typename unfold_type(TExecutable::apply)::returnType;
        using input_t = typename unfold_type(TExecutable::apply)::inputType;
        
//        template<IOperatorInput ... TInputTypes> /// <-- causes g++ compiler error
        template<typename ... TInputTypes>
        static
        std::vector<output_t* >* apply(size_t threadCnt, TInputTypes ... inputs){
            
            /// Check if the indexed access to input arguments (using get(input)) yields the correct object types needed for the Executable.
            /// E.g. get<PartitionedColumn<TFormat,...>(...) yields a column<TFormat>*
            /// or   get<uint64_t>(...) yields a uint64_t
            /// but  get<uint64_t[](...) yields also uint64_t
            static_assert(
              std::is_same<std::tuple<decltype(get(inputs,0))...>, input_t>::value,
              "[ERROR] Executor::apply : Indexed access to received arguments don't yield the expected object types for the Executable!");
            
            using namespace std;
            /// thread container
            auto threads = new std::vector<std::thread*>();
            /// result container
            auto results = new std::vector<output_t* >(threadCnt);
            
//            std::cout << "===== ===== New Executor: ===== =====" << std::endl;
            
            using core_t = ExecutorCore<TExecutable, input_t>;
//            const bool advancedDebugOutput = true;
            const bool advancedDebugOutput = false;
            
            if constexpr (advancedDebugOutput) {
                std::cout << "Elements inputList: \n  ";
                ((std::cout << inputs << ", "), ...);
                std::cout << " \n  ";
                ((std::cout << typestr(inputs) << ", "), ...);
                std::cout << std::endl;
            
                
                std::cout << "Input types after decay: \n  ";
                ((std::cout << typestr<typename std::decay<decltype(inputs)>::type>() << ", "), ...);
                std::cout << std::endl;
                
                output_t** output__ = &(results->at(0));
                
                std::cout << "Is the Core::apply Function invocable?: " << std::boolalpha;
                std::cout << std::__is_invocable<
                  decltype(core_t::apply),
                  decltype(output__),
                  decltype(get(inputs,0))...
                  >::value << std::endl;
                cout << "FunctionType of the Core:\n  " << Typestr(core_t::apply) << endl;
                cout << "Type of the Core Output Buffer:\n  " << typestr(output__) << endl;
                cout << "Types of Inputs:\n  " << typestr(inputs...) << endl;
                cout << "Types of Inputs after indexed access (get(inputs)):\n  ";
                ((cout << type_str<decltype(get(inputs, 0))>::apply() << ", "), ...);
                cout << endl;
                
                return nullptr; /// will result in an error
            } else {
    
                /// spawn threads
                for (size_t threadIdx = 0; threadIdx < threadCnt; ++ threadIdx) {
//                    std::cout << "spawn thread " << threadIdx << std::endl;
                    output_t ** output = &(results->at(threadIdx));
        
                    threads->push_back(
                      new std::thread(
                        /// core function
                        core_t::apply,
                        /// write address for lambda function result
                        output,
                        /// input parameters
                        get(inputs, threadIdx)...
                      ));
                }
    
                /// wait for threads
                for (auto & thread : *threads) {
                    thread->join();
                    delete thread;
                }
    
                /// finish
                return results;
            }
        }
    };
    
    template<ConcurrentType Tctype, IExecutable TExecutable>
    class AnotherExecutor : public Executable {
        template<IOperatorInput ... TInputTypes>
        static
        std::vector<typename TExecutable::outputType*>*
        apply(size_t threadCnt, TInputTypes...inputs){}
        
    };
    
    template<IExecutable TExecutable>
    class AnotherExecutor<ConcurrentType::STD_THREADS, TExecutable> : public Executable {
        
        using TIn = typename TExecutable::inputType;
        using TOut = typename TExecutable::outputType;
        
        /// @todo: ...
        using TOut2 = typename TExecutable::outputType;
//        using TOut2 = OutputTypes<decltype(TExecutable::apply(new uint64_t()))>;
//        using TOut2 = OutputTypes<decltype(TExecutable::apply)>;
        
        
      public:
        template<typename ... TInputTypes>
        static
        std::vector<TOut2*>*
        apply(size_t threadCnt, TInputTypes...inputs){
            static_assert(
              std::is_same<InputTypes<TInputTypes...>, typename TExecutable::inputType>::value,
              "Error in Executor::apply(...) : Given Parameter Types do not match the Input Types of the given operator!"
              );
            /// thread container
            auto threads = new std::vector<std::thread*>();
            /// result container
            auto results = new std::vector< typename TExecutable::outputType* >(threadCnt);

            std::cout << "===== ===== New Executor: ===== =====" << std::endl;
            /// spawn threads
            for(size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
                typename TExecutable::outputType** output = &(results->at(threadIdx));
                
                threads->push_back(new std::thread(
                    /// lambda function
//                    __ExecutorLambda<TExecutable, TOut<TOutputTypes...>, TIn<TInputTypes...>>::apply,

                    ExecutorCore<
                      TExecutable,
                      decltype(OutputTypes(TExecutable::apply(get(inputs, threadIdx)...))),
                      decltype(InputTypes(get(inputs, threadIdx)...))
                      >::apply,

                    /// write address for lambda function result
                    output,
                    /// input parameters
                    get(inputs, threadIdx)...
                ));
            }

            /// wait for threads
            for(auto& thread : *threads){
                thread->join();
                delete thread;
            }

            /// finish
            return results;
        }
    };
    
    template<ConcurrentType Tctype, IExecutable TExecutable>
    class AnotherExecutor2 : public Executable {
      public:
        static
        std::vector<typename TExecutable::outputType*>*
        apply(size_t, typename TExecutable::inputType){}
        
    };
    /*
    template<IExecutable TExecutable>
    class AnotherExecutor2<ConcurrentType::STD_THREADS, TExecutable> : public Executable {
        
        template<typename ... InputTypes>
        using TIn = typename TExecutable::inputType;
        using TOut = typename TExecutable::outputType;
        
      public:
        static
        std::vector<TOut*>*
        apply(size_t threadCnt, TIn input){
            /// thread container
            auto threads = new std::vector<std::thread*>();
            /// result container
            auto results = new std::vector< TOut* >(threadCnt);

            std::cout << "===== ===== New Executor: ===== =====" << std::endl;
            /// spawn threads
            for(size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
                typename TExecutable::outputType** output = &(results->at(threadIdx));
                
                threads->push_back(new std::thread(
                    /// lambda function
//                    __ExecutorLambda<TExecutable, TOut<TOutputTypes...>, TIn<TInputTypes...>>::apply,

                    __ExecutorLambda<
                      TExecutable,
                      decltype(OutputTypes(TExecutable::apply(get(inputs, threadIdx)...))),
                      decltype(InputTypes(get(inputs, threadIdx)...))
                      >::apply,

                    /// write address for lambda function result
                    output,
                    /// input parameters
                    get(inputs, threadIdx)...
                ));
            }

            /// wait for threads
            for(auto& thread : *threads){
                thread->join();
                delete thread;
            }

            /// finish
            return results;
        }
    };
    /**/
    
//    template<typename ConcurrentType, ConcurrentType T>
//    struct type_str<ConcurrentType>{
//        static string apply(){
//            return "ConcurrentType";
//        }
//    };
    
//    template<>
//    struct type_str2<ConcurrentType::STD_THREADS>{
//
//    };
    
    template<ConcurrentType TCtype, typename...Args>
    struct type_str<Executor<TCtype, Args...>>{
        static string apply(){
            return "Executor<" + demangle(typeid(TCtype).name()) + ", " + type_str<Args...>::apply() + ">";
        }
    };
    
    
} // namespace

#endif //MORPHSTORE_INCLUDE_CORE_VIRTUAL_EXECUTOR_H
