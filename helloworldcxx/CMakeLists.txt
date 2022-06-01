cmake_minimum_required(VERSION 3.10)
set(P3D_HEADER_PATH "/Library/Developer/Panda3D/include/") 
set(P3D_LIB_PATH "/Library/Developer/Panda3D/lib/") # might use "export DYLD_LIBRARY_PATH=/Library/Developer/Panda3D/lib" to link at runtime
set(P3D_LIBS "p3framework" "panda" "pandafx" "pandaexpress" "p3dtoolconfig" "p3dtool" "p3direct") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(TARGET_NAME "Panda3DCppHello.exe")

# set the project name
project(Panda3DCppHello)

include_directories(${P3D_HEADER_PATH}) # set header search path
link_directories(${P3D_LIB_PATH}) # set library search path

add_executable(${TARGET_NAME} main.cxx)
target_link_libraries(${TARGET_NAME} ${P3D_LIBS}) 

# copy runtime dynamic libs
#add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
#  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${TARGET_NAME}> $<TARGET_FILE_DIR:${TARGET_NAME}>
#  COMMAND_EXPAND_LISTS
#  )
