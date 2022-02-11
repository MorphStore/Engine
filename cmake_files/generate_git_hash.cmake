function(generate_git_hash)

    if(NOT DEFINED MorphStoreRoot)
        message( FATAL_ERROR "MorphStoreRoot is not set. generate_git_hash() can not be executed.")
    endif()

    set(GIT_BRANCH unknown PARENT_SCOPE)
    set(GIT_COMMIT_HASH unknown PARENT_SCOPE)
    # Get the current working branch

    execute_process(
            COMMAND git rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${MorphStoreRoot}
            OUTPUT_VARIABLE TMP_GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(DEFINED TMP_GIT_BRANCH)
        set(GIT_BRANCH ${TMP_GIT_BRANCH} PARENT_SCOPE)
    endif()
    # Get the latest abbreviated commit hash of the working branch
    execute_process(
            COMMAND git log -1 --format=%h
            WORKING_DIRECTORY ${MorphStoreRoot}
            OUTPUT_VARIABLE TMP_GIT_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(DEFINED TMP_GIT_COMMIT_HASH)
        set(GIT_COMMIT_HASH ${TMP_GIT_COMMIT_HASH} PARENT_SCOPE)
    endif()

    add_compile_definitions(GIT_COMMIT_HASH=${GIT_COMMIT_HASH})
    add_compile_definitions(GIT_BRANCH=${GIT_BRANCH})

    ####### This copies the current git branch and commit hash into a header file for usage inside c++
    configure_file(
       ${MorphStoreRoot}/include/core/utils/cmake_template.h.in
       ${CMAKE_BINARY_DIR}/generated/cmake_template.h
    )
endfunction()

