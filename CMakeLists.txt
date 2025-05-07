cmake_minimum_required(VERSION 3.30)
project(MandelbrotProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Find packages using vcpkg
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System Network )
find_package(TGUI 1 REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic -fPIE)
elseif(MSVC)
    add_compile_options(/W0 /MP /std:c++latest)
endif()

# Source files
set(CPP_SOURCES
        include/kernelCodeStr.h
        include/ClassImplementation/Macros.h
        src/main.cpp
        src/kernelCodeStr.cpp
        include/ClassImplementation/JuliaTimelapse.h
        include/ClassImplementation/FractalInteraction.h
        include/ClassImplementation/Fractals/MandelbrotRendering.cuh
        include/ClassImplementation/Fractals/JuliaRendering.cuh
        include/ClassImplementation/CpuFallback.h
        include/ClassImplementation/CustomFormulaHandling.h
        include/ClassImplementation/PaletteHandler.h
        include/ClassImplementation/IterationPath.h

)

set(CUDA_SOURCES
        src/CUDA_ComputationFunctions.cu
        src/benchmark.cu
        src/fractals/mandelbrot.cu
        src/fractals/julia.cu
        src/FractalClass.cu
        include/fractals/custom.cuh
        include/ClassImplementation/FractalClass.cuh
        include/ClassImplementation/Processing.cu


)

set(SOURCE_FONT_RESOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fonts)

if(EXISTS ${SOURCE_FONT_RESOURCE_DIR})
    message(STATUS "Found fonts directory: ${SOURCE_FONT_RESOURCE_DIR}")
    set(DESTINATION_DIR ${CMAKE_BINARY_DIR})
    message(STATUS "Build directory (CMAKE_BINARY_DIR): ${DESTINATION_DIR}")
    file(COPY ${SOURCE_FONT_RESOURCE_DIR} DESTINATION ${DESTINATION_DIR})
    message(STATUS "Configured copying ${SOURCE_FONT_RESOURCE_DIR} to ${DESTINATION_DIR}")
else()
    message(WARNING "Fonts directory not found at ${SOURCE_FONT_RESOURCE_DIR}. Fonts will not be copied.")
endif()

set(SOURCE_IMAGES_RESOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Images)

if(EXISTS ${SOURCE_IMAGES_RESOURCE_DIR})
    message(STATUS "Found images directory: ${SOURCE_IMAGES_RESOURCE_DIR}")
    set(DESTINATION_DIR ${CMAKE_BINARY_DIR})
    message(STATUS "Build directory (CMAKE_BINARY_DIR): ${DESTINATION_DIR}")
    file(COPY ${SOURCE_IMAGES_RESOURCE_DIR} DESTINATION ${DESTINATION_DIR})
    message(STATUS "Configured copying ${SOURCE_IMAGES_RESOURCE_DIR} to ${DESTINATION_DIR}")
else()
    message(WARNING "Images directory not found at ${SOURCE_IMAGES_RESOURCE_DIR}. Images will not be copied.")
endif()

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

find_library(CUDA_DRIVER_LIBRARY cuda)
if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "Could not find CUDA driver library")
endif()

# Find the NVRTC library for runtime compilation
find_library(NVRTC_LIBRARY nvrtc PATHS /opt/cuda/lib64 /usr/local/cuda/lib64)
if(NOT NVRTC_LIBRARY)
    message(FATAL_ERROR "Could not find NVRTC library")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fractals
        /opt/cuda/include
        ${TGUI_INCLUDE_DIRS}
        ${SFML_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        SFML::Graphics
        SFML::Window
        SFML::System
        SFML::Network
        TGUI::TGUI
        ${CUDA_DRIVER_LIBRARY}  # Link CUDA driver library
        ${NVRTC_LIBRARY}        # Link NVRTC library
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY "Shared"
        CUDA_ARCHITECTURES "native"
)


target_compile_options(${PROJECT_NAME} PRIVATE
        # C++ Host Compiler Flags (for GCC/Clang)
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-ffast-math>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-ffast-math>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-march=native>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-march=native>

        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/Ox>
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/fp:fast>
        $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Release>>:-use_fast_math>
)
