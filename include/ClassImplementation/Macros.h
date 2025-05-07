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


#define INIT_CU_RESOURCES \
  if (!custom_formula) {  \
    set_context(context_type::NVRTC); \
  } \
  CU_SAFE_CALL(cuCtxSetCurrent(ctx)); \
  nvrtcProgram prog = nullptr; \
  nvrtcResult compileResult{}; \
  std::string lowered_kernel_name_float_str; \
  std::string lowered_kernel_name_double_str; \
  std::string lowered_kernel_name_ssaa_str;

// This macro takes a string as an argument an uses it to define variable names and other stuff, just basic nvrtc stuff
#define COMPILE_PROGRAM(fractal_name_str) \
    try { \
        const std::ifstream header_file("../include/fractals/custom.cuh"); \
        if (!header_file.is_open()) { \
            std::cerr << "Error: Could not open header file: ../include/fractals/custom.cuh\n"; \
            if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
            if (module_loaded) { CU_SAFE_CALL(cuModuleUnload(module)); module = nullptr; module_loaded = false; } \
            return ""; \
        } \
        std::stringstream buffer; \
        buffer << header_file.rdbuf(); \
        const std::string header_content = buffer.str(); \
        const char *header_data = header_content.c_str(); \
        constexpr char *header_name = "custom.cuh"; \
        \
        if (prog) { \
             NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; \
        } \
        NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), ("custom_" fractal_name_str ".cu"), 1, &header_data, &header_name)); \
        \
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, ("fractal_rendering_" fractal_name_str "<float>"))); \
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, ("fractal_rendering_" fractal_name_str "<double>"))); \
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "ANTIALIASING_SSAA4")); \
        \
        std::vector<const char*> compile_options; \
        compile_options.push_back("--gpu-architecture=compute_75"); \
        const char** opts = compile_options.data(); \
        int num_opts = compile_options.size(); \
        \
        compileResult = nvrtcCompileProgram(prog, num_opts, opts); \
        \
        size_t logSize = 0; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize)); } \
        std::string log; \
        if (logSize > 1 && prog) { \
            log.resize(logSize); \
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0])); \
        } \
        \
        if (compileResult != NVRTC_SUCCESS) { \
            std::cerr << "---------------------\n"; \
            std::cerr << "NVRTC Compilation Failed for " fractal_name_str ":\n"; \
            std::cerr << "Result Code: " << nvrtcGetErrorString(compileResult) << "\n"; \
            std::cerr << "---------------------\n"; \
            std::cerr << "Compilation Log:\n"; \
            std::cerr << log << std::endl; \
            std::cerr << "---------------------\n\n"; \
            if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; }          \
            if(!module_loaded) { set_context(context_type::CUDA); }\
            return log; \
        } else { \
            std::cout << "NVRTC Compilation Succeeded for " fractal_name_str ".\n"; \
            if (!log.empty() && log.length() > 1) { \
                std::cout << "---------------------\n"; \
                std::cout << "Compilation Log (Warnings/Info):\n"; \
                std::cout << log << std::endl; \
                std::cout << "---------------------\n\n"; \
            } \
        } \
        \
        const char* lowered_name_float_ptr = nullptr; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, ("fractal_rendering_" fractal_name_str "<float>"), &lowered_name_float_ptr)); } \
        if (!lowered_name_float_ptr) { \
            std::cerr << "Error: Could not get lowered name for fractal_rendering_" fractal_name_str "<float>\n"; \
            if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
            return ""; \
        } \
        lowered_kernel_name_float_str = lowered_name_float_ptr; \
        \
        const char* lowered_name_double_ptr = nullptr; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, ("fractal_rendering_" fractal_name_str "<double>"), &lowered_name_double_ptr)); } \
        if (!lowered_name_double_ptr) { \
            std::cerr << "Error: Could not get lowered name for fractal_rendering_" fractal_name_str "<double>\n"; \
            if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
            return ""; \
        } \
        lowered_kernel_name_double_str = lowered_name_double_ptr; \
        \
        const char* lowered_name_ssaa_ptr = nullptr; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "ANTIALIASING_SSAA4", &lowered_name_ssaa_ptr)); } \
        if (!lowered_name_ssaa_ptr) { \
            std::cerr << "Error: Could not get lowered name for ANTIALIASING_SSAA4\n"; \
            if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
            return ""; \
        } \
        lowered_kernel_name_ssaa_str = lowered_name_ssaa_ptr; \
        \
        size_t ptxSize = 0; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize)); } \
        std::vector<char> ptx; \
        if (ptxSize > 0 && prog) { \
            ptx.resize(ptxSize); \
            NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data())); \
        } else if (ptxSize == 0 && prog) { \
            std::cerr << "Warning: NVRTC compilation produced 0-byte PTX.\n"; \
        } \
        \
        if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
        \
        if (module_loaded) { \
            CU_SAFE_CALL(cuModuleUnload(module)); \
            module = nullptr; \
            module_loaded = false; \
        } \
        kernelFloat = nullptr; \
        kernelDouble = nullptr; \
        kernelAntialiasing = nullptr; \
        \
        if (ptx.empty()) { \
            std::cerr << "Error: PTX generation failed or resulted in empty PTX. Cannot load module.\n"; \
            return ""; \
        } \
        CU_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0)); \
        module_loaded = true; \
        \
        if (!module) { \
             std::cerr << "Internal Error: Module is null after successful loading attempt.\n"; \
             return ""; \
        } \
        \
        if (lowered_kernel_name_float_str.empty()) { \
             std::cerr << "Internal Error: Float lowered name string is empty.\n"; \
             if (module_loaded) { cuModuleUnload(module); module = nullptr; module_loaded = false; } \
             return ""; \
        } \
        CU_SAFE_CALL(cuModuleGetFunction(&kernelFloat, module, lowered_kernel_name_float_str.c_str())); \
        \
        if (lowered_kernel_name_double_str.empty()) { \
             std::cerr << "Internal Error: Double lowered name string is empty.\n"; \
             if (module_loaded) { cuModuleUnload(module); module = nullptr; module_loaded = false; } \
             return ""; \
        } \
        CU_SAFE_CALL(cuModuleGetFunction(&kernelDouble, module, lowered_kernel_name_double_str.c_str())); \
        \
        if (lowered_kernel_name_ssaa_str.empty()) { \
             std::cerr << "Internal Error: SSAA lowered name string is empty.\n"; \
             if (module_loaded) { cuModuleUnload(module); module = nullptr; module_loaded = false; } \
             return ""; \
        } \
        CU_SAFE_CALL(cuModuleGetFunction(&kernelAntialiasing, module, lowered_kernel_name_ssaa_str.c_str())); \
        \
        std::cout << "NVRTC kernel and CUDA module loaded successfully for " fractal_name_str ".\n"; \
        custom_formula = true; \
        \
    } catch (const std::exception& e) { \
        std::cerr << "Caught Exception during NVRTC/CUDA setup for " fractal_name_str ": " << e.what() << std::endl; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
        if (module_loaded) { CU_SAFE_CALL(cuModuleUnload(module)); module = nullptr; module_loaded = false; } \
        kernelFloat = nullptr; kernelDouble = nullptr; kernelAntialiasing = nullptr; \
        lowered_kernel_name_float_str.clear(); lowered_kernel_name_double_str.clear(); lowered_kernel_name_ssaa_str.clear(); \
        return ""; \
    } catch (...) { \
        std::cerr << "Caught Unknown Exception during NVRTC/CUDA setup for " fractal_name_str << std::endl; \
        if (prog) { NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); prog = nullptr; } \
        if (module_loaded) { CU_SAFE_CALL(cuModuleUnload(module)); module = nullptr; module_loaded = false; } \
        kernelFloat = nullptr; kernelDouble = nullptr; kernelAntialiasing = nullptr; \
        lowered_kernel_name_float_str.clear(); lowered_kernel_name_double_str.clear(); lowered_kernel_name_ssaa_str.clear(); \
        return ""; \
    }

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