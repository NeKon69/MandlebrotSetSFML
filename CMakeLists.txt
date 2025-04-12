cmake_minimum_required(VERSION 3.20)
project(MandelbrotProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(TGUI_DIR "$ENV{HOME}/vcpkg/installed/x64-linux/share/tgui")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

# Tell CMake to use vcpkg
set(CMAKE_TOOLCHAIN_FILE "$ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Find packages using vcpkg
find_package(SFML 3 COMPONENTS Graphics Window System Network REQUIRED)
find_package(TGUI 1 REQUIRED)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic -fPIE)
elseif(MSVC)
    add_compile_options(/W4 /MP /std:c++latest)
endif()

# Source files
set(CPP_SOURCES
        src/main.cpp
        # Add other .cpp files if any
)

set(CUDA_SOURCES
        src/CUDA_ComputationFunctions.cu
        src/benchmark.cu
        src/fractals/mandelbrot.cu # Assumes this defines the mandelbrot specific kernel/logic
        src/fractals/julia.cu     # Assumes this defines the julia specific kernel/logic
        src/FractalClass.cu       # Contains the *implementation* of FractalBase
)

set(SOURCE_RESOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fonts)

# Check if the source directory exists
if(EXISTS ${SOURCE_RESOURCE_DIR})
    message(STATUS "Found fonts directory: ${SOURCE_RESOURCE_DIR}")

    # Use CMAKE_BINARY_DIR which points to the root of the build directory (e.g., cmake-build-debug)
    set(DESTINATION_DIR ${CMAKE_BINARY_DIR})
    message(STATUS "Build directory (CMAKE_BINARY_DIR): ${DESTINATION_DIR}")

    # Copy the source directory TO the destination directory.
    # This will create e.g. cmake-build-debug/fonts/
    file(COPY ${SOURCE_RESOURCE_DIR} DESTINATION ${DESTINATION_DIR})

    message(STATUS "Configured copying ${SOURCE_RESOURCE_DIR} to ${DESTINATION_DIR}")
else()
    message(WARNING "Fonts directory not found at ${SOURCE_RESOURCE_DIR}. Fonts will not be copied.")
endif()

# Mark .cu files as CUDA files (Good practice, though add_executable often infers it)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Create executable
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# Add CUDA include directory (already present)
# include_directories(/opt/cuda/include) # Use target_include_directories instead for modern CMake

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include # Where FractalClass.cuh should be
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fractals # Where mandelbrot.cuh/julia.cuh might be
        /opt/cuda/include # Add CUDA includes as SYSTEM includes if needed, or let find_package(CUDA) handle it
        ${TGUI_INCLUDE_DIRS} # Often provided by find_package
        ${SFML_INCLUDE_DIR}  # Often provided by find_package
)

# Link libraries (already present and looks okay)
target_link_libraries(${PROJECT_NAME} PRIVATE
        SFML::Graphics
        SFML::Window
        SFML::System
        SFML::Network
        TGUI::TGUI
        Threads::Threads
        # CUDAToolkit::cudart # Modern CMake prefers using targets from find_package(CUDAToolkit) if available
        X11 # May or may not be needed depending on SFML/TGUI setup
        dl  # May or may not be needed
)

# CUDA settings (already present and look okay)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON # Usually needed with SEPARABLE_COMPILATION
        CUDA_RUNTIME_LIBRARY "Shared"  # Or Static if preferred
        CUDA_ARCHITECTURES "native"    # Or specific ones like "75" instead of using CMAKE_CUDA_FLAGS
)