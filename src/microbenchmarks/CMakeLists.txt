add_executable( iterator_variants iterator_variants.cpp )
target_compile_options( iterator_variants PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( iterator_variants PRIVATE "-ldl" )