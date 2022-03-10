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


#ifndef MORPHSTORE_INCLUDE_CORE_UTILS_TYPE_TRAITS_H
#define MORPHSTORE_INCLUDE_CORE_UTILS_TYPE_TRAITS_H

#include <stdlibs>
#include "typestr.h"


namespace morphstore {
    /// static constexpr condition check to always deliver false result
    template< typename...Types >
    struct always_false : std::false_type {
    };
    
    

    /**
     * @Debug_Information Given template parameter is a <b>list of parameter</b>.
     * @description This class is a tool to deduce various types from a given type or list of types.
     * @tparam TypeList... A Template Parameter Pack.
     * @returns typeList = std::tuple<List...>
     */
    template< typename...TypeList >
    struct unfold_type{
        /// @description The passed Template parameter pack stored inside a std::tuple<TypeList...>. ( from <i>TypeList...</i> )
        using typeList = std::tuple<TypeList...>;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>basic type</b>.
     * @tparam BasicType A very plain type.
     */
    template< typename BasicType >
    struct unfold_type<BasicType>{
        /// @description The passed type. ( <i>basic</i> )
        using type = BasicType;
        /// @description Base type without any modifiers. (from <i>basic</i>)
        using baseType = BasicType;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const type</b>.
     * @tparam ConstType A const type.
     */
    template< typename ConstType >
    struct unfold_type<const ConstType>{
        /// @description The passed type. ( <i>const object</i> )
        using type = const ConstType;
        /// @description Base type without any modifiers. ( from <i>const object</i> )
        using baseType = ConstType;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>pointer to object</b>.
     * @tparam Pointer A pointer to an object.
     */
    template< typename Pointer >
    struct unfold_type<Pointer*>{
        /// @description The passed type. ( <i>pointer to object</i> )
        using type = Pointer*;
        /// @description Base type without any modifiers. ( from <i>pointer to object</i> )
        using baseType = Pointer;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const pointer to object</b>.
     * @tparam ConstPointer A const pointer to an object.
     */
    template< typename ConstPointer >
    struct unfold_type<ConstPointer * const>{
        /// @description The passed type. ( <i>const pointer to object</i> )
        using type = ConstPointer * const;
        /// @description Base type without any modifiers. ( from <i>const pointer to object</i> )
        using baseType = ConstPointer;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>pointer to const object</b>.
     * @tparam Const_Pointer A pointer to a const object
     */
    template< typename Const_Pointer >
    struct unfold_type<const Const_Pointer *>{
        /// @description The passed type. ( <i>pointer to const object</i> )
        using type = const Const_Pointer *;
        /// @description Base type without any modifiers. ( from <i>pointer to const object</i> )
        using baseType = Const_Pointer;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const pointer to const object</b>.
     * @tparam Const_ConstPointer A const pointer to a const object.
     */
    template< typename Const_ConstPointer >
    struct unfold_type<const Const_ConstPointer * const>{
        /// @description The passed type. ( <i>const pointer to const object</i> )
        using type = const Const_ConstPointer * const;
        /// @description Base type without any modifiers. ( from <i>const pointer to const object</i> )
        using baseType = Const_ConstPointer;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>reference</b>.
     * @tparam Reference A reference.
     */
    template< typename Reference >
    struct unfold_type<Reference &>{
        /// @description The passed type. ( <i>reference</i> )
        using type = Reference &;
        /// @description Base type without any modifiers. ( from <i>reference</i> )
        using baseType = Reference;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>rvalue</b>.
     * @tparam RValue A rvalue.
     */
    template< typename RValue >
    struct unfold_type<RValue &&>{
        /// @description The passed type. ( <i>rvalue</i> )
        using type = RValue &&;
        /// @description Base type without any modifiers. ( from <i>rvalue</i> )
        using baseType = RValue;
    };
    
    
    // ============================================= templated types =====================================================//
    /**
     * @Debug_Information Given template parameter is a <b>templated type</b>.
     * @tparam TemplateType A templated type.
     * @tparam Args A variadic set of types.
     */
    template< template<typename...> typename TemplateType, typename...Args >
    struct unfold_type<TemplateType<Args...>>{
        /// @description The passed type. ( <i>templated type</i> )
        using type = TemplateType<Args...>;
        /// @description Base type without any modifiers. (from <i>templated type</i>)
        using baseType = TemplateType<Args...>;
        /// @description The variadic set of template parameter types (or single type). (from <i>templated type</i>)
        using templateType = typename unfold_type<Args...>::type;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const templated type</b>.
     * @tparam TemplateType A templated type.
     * @tparam Args A variadic set of types.
     */
    template< template<typename...> typename TemplateType, typename...Args >
    struct unfold_type<const TemplateType<Args...>>{
        /// @description The passed type. ( <i>const templated type</i> )
        using type = const TemplateType<Args...>;
        /// @description Base type without any modifiers. (from <i>const templated type</i>)
        using baseType = TemplateType<Args...>;
        /// @description The variadic set of template parameter types (or single type). (from <i>const templated type</i>)
        using templateType = typename unfold_type<Args...>::type;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>pointer to templated type</b>.
     * @tparam TemplateType A templated type.
     * @tparam Args A variadic set of types.
     */
    template< template<typename...> typename TemplateType, typename...Args >
    struct unfold_type<TemplateType<Args...> *>{
        /// @description The passed type. ( <i>pointer to templated type</i> )
        using type = TemplateType<Args...> *;
        /// @description Base type without any modifiers. (from <i>pointer to templated type</i>)
        using baseType = TemplateType<Args...>;
        /// @description The variadic set of template parameter types (or single type). (from <i>pointer to templated type</i>)
        using templateType = typename unfold_type<Args...>::type;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>pointer to const templated type</b>.
     * @tparam TemplateType A templated type.
     * @tparam Args A variadic set of types.
     */
    template< template<typename...> typename TemplateType, typename...Args >
    struct unfold_type<const TemplateType<Args...> *>{
        /// @description The passed type. ( <i>pointer to const templated type</i> )
        using type = const TemplateType<Args...> *;
        /// @description Base type without any modifiers. (from <i>pointer to const templated type</i>)
        using baseType = TemplateType<Args...>;
        /// @description The variadic set of template parameter types (or single type). (from <i>pointer to const templated type</i>)
        using templateType = typename unfold_type<Args...>::type;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const pointer to templated type</b>.
     * @tparam TemplateType A templated type.
     * @tparam Args A variadic set of types.
     */
    template< template<typename...> typename TemplateType, typename...Args >
    struct unfold_type<TemplateType<Args...> * const>{
        /// @description The passed type. ( <i>const pointer to templated type</i> )
        using type = TemplateType<Args...> * const;
        /// @description Base type without any modifiers. (from <i>const pointer to templated type</i>)
        using baseType = TemplateType<Args...>;
        /// @description The variadic set of template parameter types (or single type). (from <i>const pointer to templated type</i>)
        using templateType = typename unfold_type<Args...>::type;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const pointer to const templated type</b>.
     * @tparam TemplateType A templated type.
     * @tparam Args A variadic set of types.
     */
    template< template<typename...> typename TemplateType, typename...Args >
    struct unfold_type<const TemplateType<Args...> * const>{
        /// @description The passed type. ( <i>const pointer to const templated type</i> )
        using type = const TemplateType<Args...> * const;
        /// @description Base type without any modifiers. (from <i>const pointer to const templated type</i>)
        using baseType = TemplateType<Args...>;
        /// @description The variadic set of template parameter types (or single type). (from <i>const pointer to const templated type</i>)
        using templateType = typename unfold_type<Args...>::type;
    };
    
    
    // ============================================= functions ===========================================================//
    /**
     * @Debug_Information Given template parameter is a <b>Function</b>.
     * @tparam ReturnType Return type of the function.
     * @tparam Args Parameter types of the function.
     */
    template<typename ReturnType, typename...Args>
    struct unfold_type<ReturnType (Args...) >{
        /// @description The passed type. ( <i>function</i> )
        using type = ReturnType (Args...);
        /// @description The return type of the function. ( <i>function</i> )
        using returnType = ReturnType;
        /// @description The input types of the function packed into a std::tuple.  ( <i>function</i> )
        using inputType = std::tuple<Args...>;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>Function Pointer</b>.
     * @tparam ReturnType Return type of the function.
     * @tparam Args Parameter types of the function.
     */
    template<typename ReturnType, typename...Args>
    struct unfold_type<ReturnType (*) (Args...) >{
        /// @description The passed type. ( <i>function pointer</i> )
        using type = ReturnType (*) (Args...);
        /// @description The return type of the function. ( <i>function pointer</i> )
        using returnType = ReturnType;
        /// @description The input types of the function packed into a std::tuple.  ( <i>function pointer</i> )
        using inputType = std::tuple<Args...>;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>Class Member Function</b>.
     * @tparam Class Class of the function.
     * @tparam ReturnType Return type of the function.
     * @tparam Args Parameter types of the function.
     */
    template<typename Class, typename ReturnType, typename...Args>
    struct unfold_type<ReturnType (Class::*) (Args...) >{
        /// @description The passed type. ( <i>class member function</i> )
        using type = ReturnType (Class::*) (Args...);
        using classType = Class;
        /// @description The return type of the function. ( <i>class member function</i> )
        using returnType = ReturnType;
        /// @description The input types of the function packed into a std::tuple.  ( <i>class member function</i> )
        using inputType = std::tuple<Args...>;
    };
    
    /**
     * @Debug_Information Given template parameter is a <b>const Class Member Function</b>.
     * @tparam Class Class of the function.
     * @tparam ReturnType Return type of the function.
     * @tparam Args Parameter types of the function.
     */
    template<typename Class, typename ReturnType, typename...Args>
    struct unfold_type<ReturnType (Class::*) (Args...) const >{
        /// @description The passed type. ( <i>const class member function</i> )
        using type = ReturnType (Class::*) (Args...) const;
        using classType = Class;
        /// @description The return type of the function. ( <i>const class member function</i> )
        using returnType = ReturnType;
        /// @description The input types of the function packed into a std::tuple.  ( <i>const class member function</i> )
        using inputType = std::tuple<Args...>;
    };
    
    /// does not work
    template<typename Class, typename Member>
    struct unfold_type<Member Class::* >{
        using classType = Class;
        using memberType = Member;
    };
    //template<typename Class, typename Member>
    //struct analyse_object< Member Class:: * >{
    //    using classType = Class;
    //    using memberType = Member;
    //};
    
    
    template<typename...Args>
    struct type_str<unfold_type<Args...>>{
        static string apply(){
            return "unfold_type<" + type_str<Args...>::apply() + ">";
        }
    };
    
    
    /**
     * @description Convenience macro to enable support of unfold_type for object instances, e.g. (member) variables,
     * (member) functions and static class methods.
     *
     * @remark Pro tip: If your IDE supports on the fly processing of documentation macros and a documentation tooltip
     * on hover over (like in CLion), you can hover over the class type definitions after the scope resolution operator (::)
     */
    #define unfold_type(object) unfold_type<decltype(object)>

    
    template<template<typename> typename Function, typename...Args>
    class convert_each_type_using_function {
      public:
        using type = std::tuple<typename convert_each_type_using_function<Function, Args>::type ...>;
    };
    
    template<template<typename> typename Function, typename Arg>
    class convert_each_type_using_function<Function, Arg>{
        using arg_pack = std::tuple<Arg>;
        using function_input = typename unfold_type<Function<Arg>>::inputType;
//        static_assert(std::is_same<arg_pack, input>::value, "");
        
      public:
        using type = typename unfold_type<Function<Arg>>::outputType;
    };
    
    
    
    
    template<typename...Args>
    struct pack : public std::tuple<Args...> {
        using type = std::tuple<Args...>;
        pack(Args...args) : std::tuple<Args...>(args...){}
    };
    
    template<typename...Args>
    struct pack<std::tuple<Args...>> : public std::tuple<Args...> {
        using type = std::tuple<Args...>;
        pack(std::tuple<Args...> args) : std::tuple<Args...>(args){}
    };
}


#endif //MORPHSTORE_INCLUDE_CORE_UTILS_TYPE_TRAITS_H
