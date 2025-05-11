//
// Created by progamers on 5/6/25.
//
#pragma once
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      throw std::runtime_error("ERR");                                                     \
    }                                                             \
} while(0)

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


#define MAKE_CURR_CONTEXT_OPERATION(x, y, ctx)                      \
  do {                                                              \
    if (context == context_type::CUDA){                             \
      CUDA_SAFE_CALL(x);                                            \
    }                                                               \
    else{                                                           \
      CU_SAFE_CALL(y);                                              \
    }                                                               \
} while(0)


#define COPY_PALETTE_TO_DEVICE(host, d, cu, ctx)                                                       \
  do {                                                                                          \
    if (context == context_type::CUDA){                                                         \
      CUDA_SAFE_CALL(cudaMemcpy(d, host, sizeof(Color) * paletteSize, cudaMemcpyHostToDevice));    \
    }                                                                                           \
    else{                                                                                       \
      CU_SAFE_CALL(cuMemcpyHtoD(cu, host, sizeof(Color) * paletteSize));                            \
    }                                                                                           \
} while(0)

#define ALLOCATE_ALL_IMAGE_MEMORY()                                                                                                                                                                 \
  do{                                                                                                                                                                                               \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_pixels, basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)), cuMemAlloc(&cu_d_pixels, sizeof(unsigned char) * basic_width * 2 * basic_height * 2 * 4), context);                  \
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&pixels, basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)), cuMemHostAlloc((void**)&pixels, sizeof(unsigned char) * basic_width * 2 * basic_height * 2 * 4, 0), context);      \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&ssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char)), cuMemAlloc(&CUssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char)), context);                             \
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&compressed, basic_width * basic_height * 4 * sizeof(unsigned char)), cuMemHostAlloc((void**)&compressed, basic_width * basic_height * 4 * sizeof(unsigned char), 0), context);                                                                      \
  } while(0)

#define FREE_ALL_IMAGE_MEMORY()                                                                 \
  do {                                                                                          \
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_pixels), cuMemFree(cu_d_pixels), context);           \
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(ssaa_buffer), cuMemFree(CUssaa_buffer), context);      \
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(pixels), cuMemFreeHost(pixels), context);          \
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(compressed), cuMemFreeHost(compressed), context);  \
  } while(0)

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

#define FREE_ALL_NON_IMAGE_MEMORY() \
  do {                               \
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamDestroy(stream), cuStreamDestroy(CUss), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(h_total_iterations), cuMemFreeHost(h_total_iterations), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamDestroy(dataStream), cuStreamDestroy(CUssData), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_total_iterations), cuMemFree(cu_d_total_iterations), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_palette), cuMemFree(cu_palette), context);\
  } while(0)

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

#define INIT_BASIC_VALUES \
    max_iterations = 300; \
    basic_zoom_x = 240.0; \
    basic_zoom_y = 240.0; \
    zoom_x = basic_zoom_x; \
    zoom_y = basic_zoom_y; \
    x_offset = 2.25; \
    y_offset = 1.25; \
    zoom_factor = 1.0; \
    zoom_speed = 0.1; \
    zoom_scale = 1.0; \
    maxComputationF = 50.f; \
    maxComputationD = 50.f; \
    basic_width = 800; \
    basic_height = 600; \
    width = basic_width; \
    height = basic_height;