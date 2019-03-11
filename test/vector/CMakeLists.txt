	message( "Vector/" )
    add_executable( vector_test_app main.cpp )
    target_compile_options( vector_test_app PRIVATE
#            -Werror
#            -Wall
#            -Wextra
#            -pedantic
            -fstack-protector-all
            -msse4.2 -mavx2
            $<$<CONFIG:DEBUG>:-DDEBUG> )
    target_link_libraries( vector_test_app PRIVATE "-ldl" )
