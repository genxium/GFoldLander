cmake_minimum_required(VERSION 3.18)
set(P3D_HEADER_PATH "/Library/Developer/Panda3D/include/") 
set(P3D_BULLET_HEADER_PATH "/Library/Developer/Panda3D/thirdparty/darwin-libs-a/bullet/include/") 
set(P3D_DLL_PATH "/Library/Developer/Panda3D/lib/") # might use "export DYLD_LIBRARY_PATH=/Library/Developer/Panda3D/lib" to link at runtime
set(P3D_BULLET_LIB_PATH "/Library/Developer/Panda3D/thirdparty/darwin-libs-a/bullet/lib/") 
set(P3D_DLL "p3framework" "panda" "pandafx" "pandaexpress" "p3dtoolconfig" "p3dtool" "p3direct" "pandabullet") 
set(P3D_DLL_FILES "libp3framework" "libpanda" "libpandafx" "libpandaexpress" "libp3dtoolconfig" "libp3dtool" "libp3direct" "libpandabullet") 
set(P3D_STATIC "BulletCollision" "BulletDynamics" "LinearMath") 
set(P3D_STATIC_FILES "libBulletCollision" "libBulletDynamics" "libLinearMath") 

# [WARNING] The minimal download/install of Panda3D only provides the "Release dlls" for building and running
set(CMAKE_BUILD_TYPE Debug) # Such that the built target is contained in a folder, then run "cmake --build . --config Release" such that the target executable file is built using "Release dlls" and contained in a folder "${CMAKE_BINARY_DIR}\Release" in Windows

set(DLL_SUFFIX ".dylib") 
set(STATIC_SUFFIX ".a") 
set(RT_PATH ${CMAKE_BINARY_DIR}) 

IF(APPLE)
        message(STATUS ">>> On APPLE, resetting rpath...")
		# Reference https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
		set(CMAKE_SKIP_BUILD_RPATH TRUE)
		set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
		set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
ELSEIF(WIN32)
        message(STATUS ">>> On Windows, resetting header and lib paths...")
        set(P3D_HEADER_PATH "C:\\Panda3D-1.10.11-x64\\include\\")  
        set(P3D_DLL_PATH "C:\\Panda3D-1.10.11-x64\\lib\\") 
        set(P3D_BULLET_HEADER_PATH "C:\\Panda3D-1.10.11-x64\\thirdparty\\win-libs-vc14-x64\\bullet\\include\\") 
        set(P3D_BULLET_LIB_PATH "C:\\Panda3D-1.10.11-x64\\thirdparty\\win-libs-vc14-x64\\bullet\\lib\\")  
        set(P3D_DLL ${P3D_DLL_FILES}) 
        set(P3D_STATIC_FILES P3D_STATIC) # Very interesting nomenclature for WIN32 
		set(DLL_SUFFIX ".lib") 
		set(STATIC_SUFFIX ".lib") 
		set(RT_PATH ${CMAKE_BINARY_DIR}/Release) 
ENDIF()
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME "Panda3DCppHello")

# set the project name
project(Panda3DCppHello)

include_directories(${P3D_BULLET_HEADER_PATH} ${P3D_HEADER_PATH}) # set header search path
link_directories(${P3D_BULLET_LIB_PATH} ${P3D_DLL_PATH}) # set library search path

add_executable(${TARGET_NAME} main.cxx)
target_link_libraries(${TARGET_NAME} ${P3D_STATIC} ${P3D_DLL}) 

IF(APPLE)
		# Debugging reference https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
		message(STATUS ">>> On APPLE, Adding @rpath=${P3D_DLL_PATH}")
		# Can be verified by "otool -l ./Panda3DCppHello | grep LC_RPATH -A2"
		add_custom_command(
			TARGET ${TARGET_NAME} POST_BUILD 
			COMMAND ${CMAKE_INSTALL_NAME_TOOL} -add_rpath ${P3D_DLL_PATH} $<TARGET_FILE:${TARGET_NAME}>
		)
		# Fix the incorrectly set loader paths, can be viewed by "otool -l ./Panda3DCppHello" and check each "LC_LOAD_DYLIB" item
		foreach( so ${P3D_DLL_FILES})
			add_custom_command(
				TARGET ${TARGET_NAME} POST_BUILD 
				COMMAND ${CMAKE_INSTALL_NAME_TOOL} -change "@loader_path/../lib/${so}.1.10${DLL_SUFFIX}" "@rpath/${so}${DLL_SUFFIX}" $<TARGET_FILE:${TARGET_NAME}>
			)
		endforeach( so )
ELSEIF(WIN32)
		# copy runtime dynamic libs
		message(STATUS ">>> On Windows, RT_PATH is ${RT_PATH}")
		foreach( so ${P3D_DLL_FILES})
				add_custom_command(
						TARGET ${TARGET_NAME} POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy_if_different ${P3D_DLL_PATH}${so}${DLL_SUFFIX} ${RT_PATH}         
				)
		endforeach( so )
ENDIF()

add_custom_command(
        TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${RT_PATH}/assets
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/assets/*.bam ${RT_PATH}/assets         
)
