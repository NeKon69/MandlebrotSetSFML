cmake_minimum_required(VERSION 3.30)
project(MandelbrotProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System Network )
find_package(TGUI 1 REQUIRED)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic -fPIE)
endif()

# Source files (Keep these)
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
        include/HardCodedVars.h
)

set(SOURCE_FONT_RESOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fonts)
if(EXISTS ${SOURCE_FONT_RESOURCE_DIR})
    message(STATUS "Found fonts directory: ${SOURCE_FONT_RESOURCE_DIR}")
    file(COPY ${SOURCE_FONT_RESOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
    message(STATUS "Configured copying ${SOURCE_FONT_RESOURCE_DIR} to ${CMAKE_BINARY_DIR}/fonts")
else()
    message(WARNING "Fonts directory not found at ${SOURCE_FONT_RESOURCE_DIR}. Fonts will not be copied.")
endif()

set(SOURCE_IMAGES_RESOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Images)
if(EXISTS ${SOURCE_IMAGES_RESOURCE_DIR})
    message(STATUS "Found images directory: ${SOURCE_IMAGES_RESOURCE_DIR}")
    file(COPY ${SOURCE_IMAGES_RESOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
    message(STATUS "Configured copying ${SOURCE_IMAGES_RESOURCE_DIR} to ${CMAKE_BINARY_DIR}/Images")
else()
    message(WARNING "Images directory not found at ${SOURCE_IMAGES_RESOURCE_DIR}. Images will not be copied.")
endif()

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fractals
        ${TGUI_INCLUDE_DIRS}
        ${SFML_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        SFML::Graphics
        SFML::Window
        SFML::System
        SFML::Network
        TGUI::TGUI
        CUDA::cudart
        CUDA::nvrtc
        CUDA::cuda_driver
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY "Shared"
)

target_compile_options(${PROJECT_NAME} PRIVATE
        # --- Keep existing GCC/Clang flags ---
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-ffast-math>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-ffast-math>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-march=native>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-march=native>

        # --- MSVC C++ specific flags ---
        $<$<CXX_COMPILER_ID:MSVC>:/W3>
        $<$<CXX_COMPILER_ID:MSVC>:/MP>
        $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/Ox>
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/fp:fast>

        # --- CUDA Compiler Flags (Specific to nvcc) ---
        $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Release>>:-use_fast_math>
)