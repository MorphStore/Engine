add_executable( storage_test storage_test.cpp )
target_compile_options( storage_test PRIVATE
                        -Werror
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( storage_test PRIVATE "-ldl" )

