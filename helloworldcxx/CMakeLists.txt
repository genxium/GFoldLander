cmake_minimum_required(VERSION 3.18)
set(P3D_HEADER_PATH "/Library/Developer/Panda3D/include/") 
set(P3D_LIB_PATH "/Library/Developer/Panda3D/lib/") # might use "export DYLD_LIBRARY_PATH=/Library/Developer/Panda3D/lib" to link at runtime
set(P3D_LIBS "p3framework" "panda" "pandafx" "pandaexpress" "p3dtoolconfig" "p3dtool" "p3direct") 

# [WARNING] The minimal download/install of Panda3D only provides the "Release dlls" for building and running
set(CMAKE_BUILD_TYPE Debug) # Such that the built target is contained in a folder, then run "cmake --build . --config Release" such that the target executable file is built using "Release dlls" and contained in a folder "${CMAKE_BINARY_DIR}\Release" in Windows

IF(WIN32)
        message(STATUS ">>> On Windows, resetting header and lib paths...")
        set(P3D_HEADER_PATH "C:\\Panda3D-1.10.11-x64\\include\\") 
        set(P3D_LIB_PATH "C:\\Panda3D-1.10.11-x64\\lib\\") 
        set(P3D_LIBS "libp3framework" "libpanda" "libpandafx" "libpandaexpress" "libp3dtoolconfig" "libp3dtool" "libp3direct") 
ENDIF()
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME "Panda3DCppHello")

# set the project name
project(Panda3DCppHello)

include_directories(${P3D_HEADER_PATH}) # set header search path
link_directories(${P3D_LIB_PATH}) # set library search path

add_executable(${TARGET_NAME} main.cxx)
target_link_libraries(${TARGET_NAME} ${P3D_LIBS}) 

# copy runtime dynamic libs
IF(WIN32)
message(STATUS ">>> binary dir is ${CMAKE_BINARY_DIR}")
foreach( so ${P3D_LIBS})
        add_custom_command(
                TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${P3D_LIB_PATH}${so}.lib ${CMAKE_BINARY_DIR}/Release         
        )
endforeach( so )
ELSE()
foreach( so ${P3D_LIBS})
        add_custom_command(
                TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${P3D_LIB_PATH}${so}.dylib ${CMAKE_BINARY_DIR}         
                COMMAND_EXPAND_LISTS
        )
endforeach( so )
ENDIF()

add_custom_command(
        TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_BINARY_DIR}/CMakeCache.txt         
        COMMAND_EXPAND_LISTS
)
