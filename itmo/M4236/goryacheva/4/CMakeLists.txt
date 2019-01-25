cmake_minimum_required(VERSION 3.5.1)
project(prefix)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(
        ${PROJECT_NAME}
        main.cpp
        prefix.cl
        cl.hpp
)

target_link_libraries (
        ${PROJECT_NAME}
        ${OpenCL_LIBRARY}
)

FILE(GLOB CL_FILES *.cl)
foreach(cl_file ${CL_FILES})
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
    )
endforeach(cl_file)