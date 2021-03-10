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
    
    template<typename Type>
    concept Is_Pointer = std::is_pointer<Type>::value;
    template<typename Type>
    concept Is_No_Pointer = !std::is_pointer<Type>::value;
    
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
    
    
    
    template<IExecutable TExecutable, IOperatorOutput ... TOutputTypes, template<typename...> class TOut, IOperatorInput ... TInputTypes, template<typename...> class TIn>
    class ExecutorCore<TExecutable, TOut<TOutputTypes...>, TIn<TInputTypes...>> {
      public:
        static void apply(TOut<TOutputTypes...> ** output, TInputTypes ... inputs){
            /// @todo
            *output = new TOut<TOutputTypes...>(TExecutable::apply(inputs...));
        }
    };
    
    template<IExecutable TExecutable, IOperatorInput ... TInputTypes, template<typename...> class TIn>
      requires std::is_same< TIn<TInputTypes...>, typename unfold_type(TExecutable::apply)::inputType >::value
    class ExecutorCore<TExecutable, TIn<TInputTypes...>> {
//        using output_t = OutputTypes<typename unfold_type(TExecutable::apply)::returnType>;
        using output_t = typename pack<typename unfold_type(TExecutable::apply)::returnType>::type;
        using input_t = typename unfold_type(TExecutable::apply)::inputType;
      public:
        
        static void apply(output_t ** output, TInputTypes ... inputs){
            *output = new output_t(TExecutable::apply(inputs...));
        }
    };
    
    template<ConcurrentType Tctype, IExecutable TExecutable, typename ... TInAndOutputTypes>
    class Executor : public Executable {
        static_assert(always_false<TInAndOutputTypes...>::value, "Unsupported types");
    };
    
    /*
    template<ConcurrentType Tctype, IExecutable TExecutable, IOperatorOutput TOutput, IOperatorInput TInput>
    class Executor<Tctype, TExecutable, TOutput, TInput> : public Executable {
      public:
        static TOutput apply(TInput input){
            return TExecutable::apply(input);
        }
    };

    template<ConcurrentType Tctype, IExecutable TExecutable, IOperatorOutput ... TOutputTypes, template<typename...> class TOut, IOperatorInput TInput>
    class Executor<Tctype, TExecutable, TOut<TOutputTypes...>, TInput> : public Executable {
      public:
        static TOut<TOutputTypes...> apply(TInput input){
            return TExecutable::apply(input);
        }
    };

    template<ConcurrentType Tctype, IExecutable TExecutable, IOperatorOutput TOutput, IOperatorInput ... TInputTypes, template<typename...> class TIn>
    class Executor<Tctype, TExecutable, TOutput, TIn<TInputTypes...>> : public Executable {
      public:
        static TOutput apply(TInputTypes ... inputs){
            return TExecutable::apply(inputs...);
        }
    };
    */
    
    /// @todo: delete??
//    template<ConcurrentType Tctype, IExecutable TExecutable, IOperatorOutput ... TOutputTypes, template<typename...> class TOut, IOperatorInput ... TInputTypes, template<typename...> class TIn>
//    class Executor<Tctype, TExecutable, TOut<TOutputTypes...>, TIn<TInputTypes...>> : public Executable {
//      public:
////        static TOut<TOutputTypes...> * apply(TInputTypes ... inputs){
////            return new TOut<TOutputTypes...>(TExecutable::apply(inputs...));
////        }
//    };
    
    
    /// std::thread variant
    template<
      IExecutable TExecutable,
      IOperatorOutput ... TOutputTypes, template<typename...> class TOut,
      IOperatorInput ... TInputTypes, template<typename...> class TIn>
    class Executor<ConcurrentType::STD_THREADS, TExecutable, TOut<TOutputTypes...>, TIn<TInputTypes...>> : public Executable {
      public:
        static
//        std::vector< TOut<TOutputTypes...>* >* apply(size_t threadCnt, typename std::remove_reference<TInputTypes>::type ... inputs){
        std::vector< TOut<TOutputTypes...>* >* apply(size_t threadCnt, TInputTypes ... inputs){
            /// thread container
            auto threads = new std::vector<std::thread*>();
            /// result container
            auto results = new std::vector< TOut<TOutputTypes...>* >(threadCnt);
            
            std::cout << "===== ===== New Executor: ===== =====" << std::endl;
            /// spawn threads
            for(size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
                TOut<TOutputTypes...>** output = &(results->at(threadIdx));
                
                /*
                std::cout << std::endl << "===== New Thread: =====" << std::endl;
                std::cout << "Output: ";
                print_type(output);
                std::cout << "Inputs:";
                unpack(print_type(inputs)...);
                std::cout << "Get from input at position " << threadIdx << ": " << std::endl;
                print_type(get(inputs, threadIdx)...);
                
                std::cout << std::endl << "Calculated output type of operator: " << typeid(decltype(OutputTypes(TExecutable::apply(get(inputs, threadIdx)...)))).name() << std::endl;
                std::cout << "Input type of operator            : " << typeid(TIn<TInputTypes...>).name() << std::endl;
                std::cout << "Calculated input type of operator : " << typeid(TIn<decltype(get(inputs, threadIdx))...>).name() << std::endl;
                std::cout << "Calculated input type of operator : " << typeid(decltype(InputTypes(get(inputs, threadIdx)...))).name() << std::endl<< std::endl;
                /**/
                
//                std::cout << typestr(threadCnt) << std::endl;
//                std::cout << typestr(inputs...) << std::endl;
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
    
    
    
    /// std::thread variant
    template< IExecutable TExecutable>
    class Executor<ConcurrentType::STD_THREADS, TExecutable> : public Executable {
      public:
//        using outType_t = OutputTypes<typename unfold_type(TExecutable::apply)::returnType>;
        using outType_t = typename pack<typename unfold_type(TExecutable::apply)::returnType>::type;
        using input_t = typename unfold_type(TExecutable::apply)::inputType;
        
//        template<IOperatorInput ... TInputTypes>
        template<typename ... TInputTypes>
        static
//        std::vector< TOut<TOutputTypes...>* >* apply(size_t threadCnt, typename std::remove_reference<TInputTypes>::type ... inputs){
        std::vector< outType_t* >* apply(size_t threadCnt, TInputTypes ... inputs){
            
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
            auto results = new std::vector< outType_t* >(threadCnt);
            
            std::cout << "===== ===== New Executor: ===== =====" << std::endl;
            
            std::cout << "Size of inputs: " << sizeof...(inputs) << std::endl;
            
            
            std::cout << "Elements inputList: ";
            ((std::cout << inputs << ", "), ...);
            std::cout << " // ";
            ((std::cout << typestr(inputs) << ", "), ...);
            std::cout << std::endl;
        
            
            std::cout << "Types after decay: ";
            ((std::cout << typestr<typename std::decay<decltype(inputs)>::type>() << ", "), ...);
            std::cout << std::endl;
            
//            using lambda = ExecutorCore<
//              TExecutable,
//              decltype(OutputTypes(TExecutable::apply(get(inputs, 0)...))),
//              decltype(InputTypes(get(inputs, 0)...))
//              >;
            using lambda = ExecutorCore<TExecutable, input_t>;
            outType_t** output__ = &(results->at(0));
//
            std::cout << "IsInvocable: " << std::boolalpha;
            std::cout << std::__is_invocable<
              decltype(lambda::apply),
              decltype(output__),
              decltype(get(inputs,0))...
              >::value << std::endl;
////
            cout << "Type of ExecutorLambda: " << Typestr(lambda::apply) << endl;
            cout << "Types of Output: " << typestr(output__) << endl;
            cout << "Types of Inputs: " << typestr(inputs...) << endl;
            cout << "Types of Inputs after get(inputs): ";
            ((cout << type_str<decltype(get(inputs, 0))>::apply() << ", "), ...);
            cout << endl;
            
//            return nullptr;/*
            /// spawn threads
            for(size_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
                std::cout << "spawn thread " << threadIdx << std::endl;
                outType_t** output = &(results->at(threadIdx));
                
                threads->push_back(new std::thread(
                    /// lambda function
//                    __ExecutorLambda<TExecutable, TOut<TOutputTypes...>, TIn<TInputTypes...>>::apply,

                    lambda::apply,
//                    ExecutorCore<
//                      TExecutable,
//                      decltype(OutputTypes(TExecutable::apply(get(inputs, threadIdx)...))),
//                      decltype(InputTypes(get(inputs, threadIdx)...))
//                      >::apply,

                    /// write address for lambda function result
                    output,
                    /// input parameters
                    get(inputs, threadIdx)...
                ));
                // debug:
//                threads->at(threadIdx)->join();
            }
            
            /// wait for threads
            for(auto& thread : *threads){
                thread->join();
                delete thread;
            }
            
            /// finish
            return results;
            /**/
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
