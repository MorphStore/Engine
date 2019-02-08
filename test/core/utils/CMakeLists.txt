add_executable( printing_test printing_test.cpp )
target_compile_options( printing_test PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( printing_test PRIVATE "-ldl" )

