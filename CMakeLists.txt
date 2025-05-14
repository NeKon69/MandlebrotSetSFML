cmake_minimum_required(VERSION 3.21)
project(MandelbrotProject LANGUAGES CXX CUDA)

# --- Basic Project Setup ---
message(STATUS "--- CMake Version: ${CMAKE_VERSION} ---")
# (Other status messages as you prefer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native") # Or specific architectures

find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System Network)
find_package(TGUI 1 REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic -fPIE)
endif()

# --- Source Files ---
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
        include/fractals/custom.cuh # This is compiled in
        include/ClassImplementation/FractalClass.cuh
        include/ClassImplementation/Processing.cu
        include/HardCodedVars.h
)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fractals
        ${TGUI_INCLUDE_DIRS}
        ${SFML_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# --- Link Libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE
        SFML::Graphics SFML::Window SFML::System
        TGUI::TGUI
        CUDA::cudart CUDA::nvrtc CUDA::cuda_driver
)

# --- Target Properties ---
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY "Shared"
)

# --- Compile Options ---
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/W3 /MP /EHsc>
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/Ox /fp:fast>
        $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CONFIG:Release>>:-O3 -use_fast_math>
)

# --- Runtime Resource Copying ---
set(RUNTIME_OUTPUT_DIR $<TARGET_FILE_DIR:${PROJECT_NAME}>)

# Copy Fonts directory
set(SOURCE_FONT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fonts)
if(EXISTS "${SOURCE_FONT_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${SOURCE_FONT_DIR}"
            "${RUNTIME_OUTPUT_DIR}/fonts"
            VERBATIM)
else()
    message(WARNING "Fonts directory not found: ${SOURCE_FONT_DIR}")
endif()

# Copy Images directory
set(SOURCE_IMAGES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Images)
if(EXISTS "${SOURCE_IMAGES_DIR}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${SOURCE_IMAGES_DIR}"
            "${RUNTIME_OUTPUT_DIR}/Images"
            VERBATIM)
else()
    message(WARNING "Images directory not found: ${SOURCE_IMAGES_DIR}")
endif()

# Copy specific custom.cuh for NVRTC runtime compilation
set(NVRTC_CUSTOM_CUH_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/fractals/custom.cuh)
if(EXISTS "${NVRTC_CUSTOM_CUH_SOURCE}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVRTC_CUSTOM_CUH_SOURCE}"
            "${RUNTIME_OUTPUT_DIR}/custom.cuh" # Copies to e.g., cmake-build-release/custom.cuh
            VERBATIM)
    message(STATUS "Configured copying NVRTC ${NVRTC_CUSTOM_CUH_SOURCE} to ${RUNTIME_OUTPUT_DIR}/custom.cuh")
else()
    message(WARNING "NVRTC custom.cuh not found: ${NVRTC_CUSTOM_CUH_SOURCE}")
endif()


# --- DLL/SO Runtime Dependency Workaround ---
set(SHARED_LIB_COPY_SUBDIR "lib") # Subdirectory relative to executable for .so/.dll
set(DEST_LIB_DIR ${RUNTIME_OUTPUT_DIR}/${SHARED_LIB_COPY_SUBDIR})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DEST_LIB_DIR}"
        VERBATIM)

function(copy_library_if_exists TARGET_NAME LIB_IMPORTED_TARGET_NAME DEST_DIR)
    if(NOT TARGET ${LIB_IMPORTED_TARGET_NAME})
        message(WARNING "  - Imported target ${LIB_IMPORTED_TARGET_NAME} not found for copying.")
        return()
    endif()
    get_target_property(LIB_TYPE ${LIB_IMPORTED_TARGET_NAME} TYPE)
    if(LIB_TYPE STREQUAL "INTERFACE_LIBRARY") # Skip interface libraries
        message(STATUS "  - ${LIB_IMPORTED_TARGET_NAME} is an INTERFACE library, nothing to copy.")
        return()
    endif()

    set(FILE_TO_COPY "")
    set(CURRENT_CONFIG "")
    if(CMAKE_CONFIGURATION_TYPES)
        set(CURRENT_CONFIG "$<CONFIG>")
    elseif(CMAKE_BUILD_TYPE)
        set(CURRENT_CONFIG "${CMAKE_BUILD_TYPE}")
    endif()

    set(PROP_QUERIES)
    if(CURRENT_CONFIG AND NOT CMAKE_CONFIGURATION_TYPES) # Single config (e.g. Ninja with CMAKE_BUILD_TYPE)
        if(WIN32)
            list(APPEND PROP_QUERIES IMPORTED_LOCATION_${CURRENT_CONFIG} LOCATION_${CURRENT_CONFIG})
        else()
            list(APPEND PROP_QUERIES LOCATION_${CURRENT_CONFIG})
        endif()
    endif()
    if(WIN32)
        list(APPEND PROP_QUERIES IMPORTED_LOCATION LOCATION) # General properties
    else()
        list(APPEND PROP_QUERIES LOCATION)
    endif()

    foreach(PROP ${PROP_QUERIES})
        get_target_property(CANDIDATE_PATH ${LIB_IMPORTED_TARGET_NAME} ${PROP})
        if(CANDIDATE_PATH AND EXISTS "${CANDIDATE_PATH}")
            set(FILE_TO_COPY "${CANDIDATE_PATH}")
            break()
        endif()
    endforeach()

    if(FILE_TO_COPY)
        get_filename_component(LIB_FILENAME "${FILE_TO_COPY}" NAME)
        get_filename_component(LIB_EXT "${FILE_TO_COPY}" EXT)
        set(IS_SHARED_LIB FALSE)
        if(WIN32 AND (LIB_EXT STREQUAL ".dll"))
            set(IS_SHARED_LIB TRUE)
        elseif((UNIX AND NOT APPLE) AND (LIB_EXT STREQUAL ".so" OR LIB_EXT MATCHES "\\.so\\..*$"))
            set(IS_SHARED_LIB TRUE)
        elseif(APPLE AND (LIB_EXT STREQUAL ".dylib"))
            set(IS_SHARED_LIB TRUE)
        endif()

        if(IS_SHARED_LIB)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FILE_TO_COPY}"
                    "${DEST_DIR}/${LIB_FILENAME}"
                    VERBATIM)
            message(STATUS "  - Will copy ${LIB_FILENAME} (from ${LIB_IMPORTED_TARGET_NAME}) to ${DEST_DIR}")
        else()
            message(STATUS "  - Skipped non-shared/DLL '${LIB_FILENAME}' (from ${LIB_IMPORTED_TARGET_NAME}). Path: ${FILE_TO_COPY}")
        endif()
    else()
        message(WARNING "  - No valid file location for ${LIB_IMPORTED_TARGET_NAME}. Build type: ${CMAKE_BUILD_TYPE}.")
    endif()
endfunction()

set(LIBS_TO_COPY
        SFML::Graphics SFML::Window SFML::System
        TGUI::TGUI
        CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
if(UNIX AND NOT APPLE AND TARGET Threads::Threads)
    list(APPEND LIBS_TO_COPY Threads::Threads)
endif()

message(STATUS "Attempting to manually copy specified libraries for ${PROJECT_NAME}:")
foreach(LIB_TARGET_NAME ${LIBS_TO_COPY})
    copy_library_if_exists(${PROJECT_NAME} ${LIB_TARGET_NAME} "${DEST_LIB_DIR}")
endforeach()

# --- RPATH Settings ---
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "$ORIGIN/${SHARED_LIB_COPY_SUBDIR}")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "@loader_path/${SHARED_LIB_COPY_SUBDIR}")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "On Windows, ensure '${SHARED_LIB_COPY_SUBDIR}' relative to executable is in PATH for DLLs.")
endif()