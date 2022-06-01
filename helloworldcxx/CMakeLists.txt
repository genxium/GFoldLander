cmake_minimum_required(VERSION 3.16)
set(P3D_HEADER_PATH "/Library/Developer/Panda3D/include") 
set(P3D_LIB_PATH "/Library/Developer/Panda3D/lib") # might use "export DYLD_LIBRARY_PATH=/Library/Developer/Panda3D/lib" to link at runtime
set(P3D_LIBS "p3framework" "panda" "pandafx" "pandaexpress" "p3dtoolconfig" "p3dtool" "p3direct") 
IF(WIN32)
        message(STATUS ">>> On Windows, resetting header and lib paths...")
        set(P3D_HEADER_PATH "C:\\Panda3D-1.10.11-x64\\include") 
        set(P3D_LIB_PATH "C:\\Panda3D-1.10.11-x64\\lib") 
        set(P3D_LIBS "libp3framework" "libpanda" "libpandafx" "libpandaexpress" "libp3dtoolconfig" "libp3dtool" "libp3direct") 
ENDIF()
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME "Panda3DCppHello")

# set the project name
project(${TARGET_NAME})

include_directories(${P3D_HEADER_PATH}) # set header search path
link_directories(${P3D_LIB_PATH}) # set library search path

add_executable(${TARGET_NAME} main.cxx)
target_link_libraries(${TARGET_NAME} ${P3D_LIBS}) 

# copy runtime dynamic libs
#add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
#  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${TARGET_NAME}> $<TARGET_FILE_DIR:${TARGET_NAME}>
#  COMMAND_EXPAND_LISTS
#  )
