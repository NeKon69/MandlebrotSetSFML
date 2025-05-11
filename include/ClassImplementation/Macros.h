//
// Created by progamers on 5/6/25.
//
#pragma once
#include "FractalClass.cuh"
#include "Macros.h"
#include "HardCodedVars.h"
#include <iostream>

/// Macro to wrap NVRTC API calls for error checking.
/// If an NVRTC call fails, it prints an error message and throws a runtime_error.
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      throw std::runtime_error("ERR");                                                     \
    }                                                             \
} while(0)

/// Macro to wrap CUDA Driver API calls for error checking.
/// If a Driver API call fails, it prints an error message and throws a runtime_error.
#define CU_SAFE_CALL(x)                                           \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      throw std::runtime_error(#x " failed with error " + std::string(msg));                                                     \
    }                                                             \
} while(0)


/// Macro to wrap CUDA Runtime API calls for error checking.
/// If a Runtime API call fails, it prints an error message and throws a runtime_error.
#define CUDA_SAFE_CALL(x) \
  do {                    \
    cudaError_t result = x;                                         \
    if (result != cudaSuccess) {                                   \
      const char *msg =  cudaGetErrorName(result);                  \
      std::cerr << "\nerror: " #x " failed with error "             \
                << msg << '\n';                                     \
      throw std::runtime_error(#x " failed with error " + std::string(msg));                                                       \
    }                                                               \
  }while(0)


/// Macro to abstract operations that differ between CUDA Runtime and Driver APIs.
/// It executes 'x' if the current context is CUDA Runtime, and 'y' if it's NVRTC (Driver API).
/// This allows the rest of the code to use a single macro call regardless of the active CUDA API.
#define MAKE_CURR_CONTEXT_OPERATION(x, y, ctx)                      \
  do {                                                              \
    if (context == context_type::CUDA){                             \
      CUDA_SAFE_CALL(x);                                            \
    }                                                               \
    else{                                                           \
      CU_SAFE_CALL(y);                                              \
    }                                                               \
} while(0)


/// Macro to copy the color palette from host to device memory.
/// Uses MAKE_CURR_CONTEXT_OPERATION to handle both CUDA Runtime and Driver APIs.
#define COPY_PALETTE_TO_DEVICE(host, d, cu, ctx)                                                       \
  do {                                                                                          \
    if (context == context_type::CUDA){                                                         \
      CUDA_SAFE_CALL(cudaMemcpy(d, host, sizeof(Color) * paletteSize, cudaMemcpyHostToDevice));    \
    }                                                                                           \
    else{                                                                                       \
      CU_SAFE_CALL(cuMemcpyHtoD(cu, host, sizeof(Color) * paletteSize));                            \
    }                                                                                           \
} while(0)

/// Macro to allocate all necessary GPU and host memory for image data.
/// This includes device buffers for the main render target (`d_pixels`/`cu_d_pixels`)
/// and the anti-aliasing buffer (`ssaa_buffer`/`CUssaa_buffer`),
/// and host (CPU) pinned memory for transferring data (`pixels`, `compressed`).
/// The allocation size for `d_pixels` and `pixels` is 2x the basic resolution in each dimension,
/// anticipating 4x SSAA rendering.
#define ALLOCATE_ALL_IMAGE_MEMORY()                                                                                                                                                                 \
  do{                                                                                                                                                                                               \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_pixels, basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)), cuMemAlloc(&cu_d_pixels, sizeof(unsigned char) * basic_width * 2 * basic_height * 2 * 4), context);                  \
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&pixels, basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)), cuMemHostAlloc((void**)&pixels, sizeof(unsigned char) * basic_width * 2 * basic_height * 2 * 4, 0), context);      \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&ssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char)), cuMemAlloc(&CUssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char)), context);                             \
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&compressed, basic_width * basic_height * 4 * sizeof(unsigned char)), cuMemHostAlloc((void**)&compressed, basic_width * basic_height * 4 * sizeof(unsigned char), 0), context);                                                                      \
  } while(0)

/// Macro to free all GPU and host memory allocated for image data.
/// Uses MAKE_CURR_CONTEXT_OPERATION to handle both CUDA Runtime and Driver APIs.
#define FREE_ALL_IMAGE_MEMORY()                                                                 \
  do {                                                                                          \
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_pixels), cuMemFree(cu_d_pixels), context);           \
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(ssaa_buffer), cuMemFree(CUssaa_buffer), context);      \
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(pixels), cuMemFreeHost(pixels), context);          \
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(compressed), cuMemFreeHost(compressed), context);  \
  } while(0)

/// Macro to allocate necessary GPU and host memory for non-image data.
/// This includes device memory for the color palette (`d_palette`/`cu_palette`)
/// and the total iteration counter (`d_total_iterations`/`cu_d_total_iterations`),
/// host pinned memory for the total iteration counter (`h_total_iterations`),
/// and CUDA streams for asynchronous operations (`stream`/`CUss`, `dataStream`/`CUssData`).
/// It also copies the initial palette data to the device.
#define ALLOCATE_ALL_NON_IMAGE_MEMORY() \
  do {                                    \
    unsigned int zero = 0; \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_palette, palette.size() * sizeof(Color)), cuMemAlloc(&cu_palette, sizeof(Color) * paletteSize), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(Color), cudaMemcpyHostToDevice), cuMemcpyHtoD(cu_palette, palette.data(), sizeof(Color) * paletteSize), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamCreate(&stream), cuStreamCreate(&CUss, 0), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_total_iterations, sizeof(unsigned int)), cuMemAlloc(&cu_d_total_iterations, sizeof(unsigned int)), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMemset(d_total_iterations, 0, sizeof(unsigned int)), cuMemcpyHtoD(cu_d_total_iterations, &zero, sizeof(unsigned int)), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&h_total_iterations , sizeof(unsigned int)), cuMemHostAlloc((void**)&h_total_iterations, sizeof(unsigned int), 0), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamCreate(&dataStream), cuStreamCreate(&CUssData, 0), context);\
  } while(0)

/// Macro to free all GPU and host memory allocated for non-image data.
/// Uses MAKE_CURR_CONTEXT_OPERATION to handle both CUDA Runtime and Driver APIs.
#define FREE_ALL_NON_IMAGE_MEMORY() \
  do {                               \
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamDestroy(stream), cuStreamDestroy(CUss), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(h_total_iterations), cuMemFreeHost(h_total_iterations), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamDestroy(dataStream), cuStreamDestroy(CUssData), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_total_iterations), cuMemFree(cu_d_total_iterations), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_palette), cuMemFree(cu_palette), context);\
  } while(0)

/// Macro to recalculate rendering parameters (resolution, zoom scale, offset)
/// based on the requested render quality (good or best).
/// When switching to 'best' quality, it doubles the resolution for 4x SSAA
/// and recalculates the offset to keep the view centered on the same complex point (if switched to different quality).
#define RECALCULATE_RENDER_PARAMS(quality_arg) \
    unsigned int old_width = width, old_height = height; \
    double new_zoom_scale_local; /* Use a local variable for the new scale */ \
    if ((quality_arg) == render_state::good) { \
        width = basic_width; \
        height = basic_height; \
        antialiasing = false; \
        new_zoom_scale_local = 1.0; \
    } else { /* render_state::best */ \
        width = basic_width * 2; \
        height = basic_height * 2; \
        antialiasing = true; \
        new_zoom_scale_local = 2.0; \
    } \
    if (width != old_width || height != old_height) { \
        /* Calculate center based on old dimensions and zoom */ \
        double center_x = x_offset + (double)old_width / (zoom_x * zoom_scale) / 2.0; \
        double center_y = y_offset + (double)old_height / (zoom_y * zoom_scale) / 2.0; \
        zoom_scale = new_zoom_scale_local; /* Update member zoom_scale */ \
        /* Recalculate offset based on new dimensions and updated zoom_scale */ \
        x_offset = center_x - (double)width / (zoom_x * zoom_scale) / 2.0; \
        y_offset = center_y - (double)height / (zoom_y * zoom_scale) / 2.0; \
    } else { \
        /* If dimensions didn't change, still update zoom_scale member */ \
        zoom_scale = new_zoom_scale_local; \
    } \
    /* These become local variables in the function scope */ \
    double render_zoom_x = zoom_x * zoom_scale; \
    double render_zoom_y = zoom_y * zoom_scale; \
    size_t len = static_cast<size_t>(width) * height * 4;

/// Macro to initialize basic fractal parameters to their default hardcoded values.
#define INIT_BASIC_VALUES \
    max_iterations = MAX_ITERATIONS; \
    basic_zoom_x = BASIC_ZOOM_X; \
    basic_zoom_y = BASIC_ZOOM_Y; \
    zoom_x = basic_zoom_x; \
    zoom_y = basic_zoom_y; \
    x_offset = BASIC_X_OFFSET; \
    y_offset = BASIC_Y_OFFSET; \
    zoom_factor = BASIC_ZOOM_FACTOR; \
    zoom_speed = BASIC_ZOOM_SPEED; \
    zoom_scale = BASIC_ZOOM_SCALE; \
    basic_width = BASIC_WIDTH; \
    basic_height = BASIC_HEIGHT; \
    width = basic_width; \
    height = basic_height;