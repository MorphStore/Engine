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


#ifndef MORPHSTORE_CORE_UTILS__TYPESTR_H
#define MORPHSTORE_CORE_UTILS__TYPESTR_H

#include <string>
#include <tuple>
#include <vector>

#ifdef __GNUG__
  #include <cstdlib>
  #include <memory>
  #include <cxxabi.h>

std::string demangle(const char* name) {
    
    int status = -4; // some arbitrary value to eliminate the compiler warning
    
    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
      abi::__cxa_demangle(name, NULL, NULL, &status),
      std::free
    };
    
    return (status==0) ? res.get() : name ;
}

#else

// does nothing if not g++
    std::string demangle(const char* name) {
        return name;
    }

#endif


namespace morphstore {
    using std::string;
    
    
    /// initial construct
    template< typename...T >
    struct type_str {};
    
    /// variadic type list
    template< typename T, typename...Next >
    struct type_str<T, Next...> {
        static string apply() {
            if constexpr(sizeof...(Next) > 0)
                return type_str<T>::apply() + ", " + type_str<Next...>::apply();
            return "[FATAL] you should never reach this area [FATAL]";
        }
    };
    
    /// no type = void
    template<>
    struct type_str<> {
        static string apply() {
            return "void";
        }
    };

/// default : unkown type
    template< typename T >
    struct type_str<T> {
        static string apply() {
//            return "[" + string(typeid(T).name()) + "]";
            return "[" + string(demangle(typeid(T).name())) + "]";
        }
    };

/// pointer
    template< typename T >
    struct type_str<T *> {
        static string apply() {
            return type_str<T>::apply() + " *";
        }
    };

/// const pointer
    template< typename T >
    struct type_str<T * const> {
        static string apply() {
            return type_str<T *>::apply() + " const";
        }
    };

/// const object
    template< typename T >
    struct type_str<const T> {
        static string apply() {
            return "const " + type_str<T>::apply();
        }
    };

/// reference
    template< typename T >
    struct type_str<T &> {
        static string apply() {
            return type_str<T>::apply() + "&";
        }
    };

/// r-value
    template< typename T >
    struct type_str<T &&> {
        static string apply() {
            return type_str<T>::apply() + "&&";
        }
    };


/// nested types
    template< template< typename... > typename T, typename...Args >
    struct type_str<T<Args...>> {
        static string apply() {
//            return "[" + string(typeid(T<Args...>).name()) + "]<" + type_str<Args...>::apply() + ">";
            return "[" + demangle(typeid(T<Args...>).name()) + "]";
        }
    };

/// std::tuple
    template< typename...Args >
    struct type_str<std::tuple<Args...>> {
        static string apply() {
            return "std::tuple<" + type_str<Args...>::apply() + ">";
        }
    };

/// std::vector
    template< typename...Args >
    struct type_str<std::vector<Args...>> {
        static string apply() {
            return "std::vector<" + type_str<Args...>::apply() + ">";
        }
    };




/// function
//template<typename Function>
//struct type_str<Function () >{
//    static string apply(){
//        return "(void) -> " + type_str<Function>::apply();
//    }
//};
    
    template< typename ReturnType, typename ... Args >
    struct type_str<ReturnType(Args...)> {
        static string apply() {
//        return "(" + type_str<Args...>::apply() + ") -> " + type_str<ReturnType>::apply();
            return type_str<ReturnType>::apply() + " (" + type_str<Args...>::apply() + ")";
        }
    };
//template<typename Function, typename ... Args>
//struct type_str<Function (Args...) const>{
//    static string apply(){
//        return "(const)(" + type_str<Args...>::apply() + ") -> " + type_str<Function>::apply();
//    }
//};

/// dereference function pointer
//template<typename Function>
//struct type_str<Function (*) () >{
//    static string apply(){
//        return "(*)" + type_str<Function ()>::apply();
//    }
//};
    template< typename ReturnType, typename ... Args >
    struct type_str<ReturnType (*)(Args...)> {
        static string apply() {
//        return "(*)" + type_str<ReturnType (Args...)>::apply();
            return type_str<ReturnType>::apply() + " (*)(" + type_str<Args...>::apply() + ")";
        }
    };


/// class member function
    template< typename Class, typename ReturnType, typename...Args >
    struct type_str<ReturnType (Class::*)(Args...)> {
        static string apply() {
//        return "(" + type_str<Class>::apply() + "::*)" + type_str<ReturnType (Args...)>::apply();
            return type_str<ReturnType>::apply() + " (" + type_str<Class>::apply() + "::*)(" +
                   type_str<Args...>::apply() + ")";
        }
    };
    
    template< typename Class, typename ReturnType, typename...Args >
    struct type_str<ReturnType (Class::*)(Args...) const> {
        static string apply() {
//        return "(" + type_str<Class>::apply() + "::* const)" + type_str<ReturnType (Args...)>::apply();
            return type_str<ReturnType>::apply() + " (" + type_str<Class>::apply() + "::* const)(" +
                   type_str<Args...>::apply() + ")";
        }
    };


/// basic types
    template<>
    struct type_str<uint64_t> {
        static string apply() {
            return string("uint64_t");
        }
    };
    
    template<>
    struct type_str<uint32_t> {
        static string apply() {
            return string("uint32_t");
        }
    };
    
    template<>
    struct type_str<uint16_t> {
        static string apply() {
            return string("uint16_t");
        }
    };
    
    template<>
    struct type_str<uint8_t> {
        static string apply() {
            return string("uint8_t");
        }
    };
    
    template<>
    struct type_str<int64_t> {
        static string apply() {
            return string("int64_t");
        }
    };
    
    template<>
    struct type_str<int32_t> {
        static string apply() {
            return string("int32_t");
        }
    };
    
    template<>
    struct type_str<int16_t> {
        static string apply() {
            return string("int16_t");
        }
    };
    
    template<>
    struct type_str<int8_t> {
        static string apply() {
            return string("int8_t");
        }
    };
    
    template<>
    struct type_str<float> {
        static string apply() {
            return string("float");
        }
    };
    
    template<>
    struct type_str<double> {
        static string apply() {
            return string("double");
        }
    };
    
    template<>
    struct type_str<char> {
        static string apply() {
            return string("char");
        }
    };
    
    template<>
    struct type_str<std::string> {
        static string apply() {
            return string("std::string");
        }
    };
    
    template<>
    struct type_str<void> {
        static string apply() {
            return string("void");
        }
    };
    
    template<>
    struct type_str<bool> {
        static string apply() {
            return string("bool");
        }
    };
    
    


/// shortcuts
    template< typename...T >
    string typestr(T&& ...t) {
        return type_str<T ...>::apply();
    }
    template< typename...T >
    string typestr(T& ...t) {
        return type_str<T ...>::apply();
    }
    /**
     * Passing by value results in copying of function objects to function pointers
     * @tparam T
     * @param t
     * @return
     */
//    template< typename...T >
//    string typestr(T ...t) {
//        return type_str<T ...>::apply();
//    }

    template< typename...T >
    string typestr() {
        return type_str<T...>::apply();
    }
    
    #define Typestr(ARGS) type_str<decltype(ARGS)>::apply()
    
}
#endif //MORPHSTORE_CORE_UTILS__TYPESTR_H
