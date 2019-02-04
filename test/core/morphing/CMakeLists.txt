add_executable( morphing_test morphing_test.cpp )
target_compile_options( morphing_test PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( morphing_test PRIVATE "-ldl" )


