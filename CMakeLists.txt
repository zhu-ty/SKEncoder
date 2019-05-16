## CMakeLists.txt for H264Encoder
# author: zhu-ty
# date: Jan 18, 2019
#

cmake_minimum_required (VERSION 3.10)
set(name "SKEncoder")
project ( ${name} )

# c++ version
set (CMAKE_CXX_STANDARD 11)

#execute_process(
#	COMMAND	git log -1  --format=%ad --date=format:%y%m%d_%H%M%S
#	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} 
#	OUTPUT_VARIABLE GIT_VERSION
#)
# MESSAGE(${GIT_VERSION})
#string (REGEX REPLACE "[\n\t\r]" "" GIT_VERSION ${GIT_VERSION})
#add_definitions( -D__GIT_VERSION__=\"${GIT_VERSION}\")




# opencv 
if ( WIN32 )
    set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
    set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
    include_directories(${WIN_HEADER_PATH})
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )

file (GLOB this_headers_h "./*.h")
file (GLOB this_headers_hpp "./*.hpp")
set (headers
	${this_headers_h}
	${this_headers_hpp}
)
file (GLOB this_headers_cpp "./*.cpp")
file (GLOB this_headers_c "./*.c")
set (sources
	${this_headers_cpp}
	${this_headers_c}
)


set(MY_HEADER_FILES
    ${headers}
)

set(MY_SOURCE_FILES 
    ${sources}
)

include_directories(
    ${PROJECT_SOURCE_DIR}
)

cuda_add_executable(${name}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
)

#message (${PROJECT_SOURCE_DIR}/lib/x64/nvcuvid.lib)

#if( WIN32 )
#    set(NVCUVID_LIB ${PROJECT_SOURCE_DIR}/lib/x64/nvcuvid.lib)
#else()
#    set(NVCUVID_LIB 
#    ${PROJECT_SOURCE_DIR}/lib/linux/stubs/x86_64/libnvcuvid.so
#    ${PROJECT_SOURCE_DIR}/lib/linux/stubs/x86_64/libnvidia-encode.so)
#endif()


target_link_libraries(${name}
    cuda
    # ${NVCUVID_LIB}
    ${OpenCV_LIBS}
)


#IF(EXISTS ${PROJECT_SOURCE_DIR}/.git)
#
#add_custom_command(TARGET ${name}
#                   PRE_LINK
#                   #COMMAND git --version
#				   COMMAND powershell "git log -1 --format='%ad' > ${CMAKE_BINARY_DIR}/test.txt"
#				#--date=format:%%y%%m%%d_%%H%%M%%S
#                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
#
#ELSE()
#
#ENDIF()

FILE(WRITE ${CMAKE_BINARY_DIR}/version.h.in
"\#define __GIT_VERSION__ \"@VERSION@\"\n"
)
IF(EXISTS ${PROJECT_SOURCE_DIR}/.git)
FILE(WRITE ${CMAKE_BINARY_DIR}/version.cmake
"EXECUTE_PROCESS(
     COMMAND git log -1 --format=%ad --date=format:%y%m%d.%H%M%S
     OUTPUT_VARIABLE VERSION
     OUTPUT_STRIP_TRAILING_WHITESPACE
 )
 CONFIGURE_FILE(\${SRC} \${DST} @ONLY)
")
ELSE()
FILE(WRITE ${CMAKE_BINARY_DIR}/version.cmake
"EXECUTE_PROCESS(
     COMMAND echo unknown
     OUTPUT_VARIABLE VERSION
     OUTPUT_STRIP_TRAILING_WHITESPACE
 )
 CONFIGURE_FILE(\${SRC} \${DST} @ONLY)
")
ENDIF()
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})
#ADD_EXECUTABLE(main main.c)
ADD_CUSTOM_TARGET(
    version
    ${CMAKE_COMMAND} -D SRC=${CMAKE_BINARY_DIR}/version.h.in
                     -D DST=${CMAKE_BINARY_DIR}/version.h
                     -P ${CMAKE_BINARY_DIR}/version.cmake
)

ADD_DEPENDENCIES(${name} version)