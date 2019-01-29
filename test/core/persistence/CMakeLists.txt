add_executable( binary_io_test binary_io_test.cpp )
target_compile_options( binary_io_test PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( binary_io_test PRIVATE "-ldl" )

