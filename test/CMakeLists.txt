add_executable( memory_test core/memory/memory_test.cpp )
target_compile_options( memory_test PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( memory_test PRIVATE "-ldl" )