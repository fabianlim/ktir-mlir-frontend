if(NOT KTDP_STAND_ALONE)
    # Prevent KTDP from building or installing targets that aren't needed.
    set(EXCLUDE_FROM_ALL YES)
    set(DISABLE_INSTALL YES)
endif()

add_subdirectory(ktdp-lsp-server)
add_subdirectory(ktdp-opt)
