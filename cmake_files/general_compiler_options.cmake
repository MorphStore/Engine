# general compiler settings, meant for all subdirectories and tests
function(set_general_compiler_options)
if(NOT BUILD_TYPE MATCHES Debug)
    add_compile_options(-pedantic)
    add_compile_options(-Wall)
endif()
    add_compile_options(-Werror)
add_compile_options(-Wextra)
add_compile_options(-Wno-ignored-attributes)
add_compile_options(-Wno-comment)
add_compile_options(-fstack-protector-all)
add_compile_options(-fno-tree-vectorize)

## enable C++20 concepts
if(${CMAKE_CXX_STANDARD} GREATER_EQUAL 20)
    add_compile_definitions(USE_CPP20_CONCEPTS)
    add_compile_options(-fconcepts)
endif()
endfunction()
