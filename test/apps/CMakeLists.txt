add_executable( operator_test operator.cpp )
FILE( REMOVE ${CMAKE_BINARY_DIR}/test/core/apps/operator_test )
target_compile_options( operator_test PRIVATE
                        -Wall
                        -Wextra
                        -pedantic
                        -fstack-protector-all
                        -msse4.2 
                        -msse2
                        -mavx2 
                        -mavx512f 
                        -mavx512pf 
                        -mavx512er 
                        -mavx512cd 
                        -mavx512vl
                        $<$<CONFIG:DEBUG>:-DDEBUG> )
target_link_libraries( operator_test PRIVATE "-ldl" )
