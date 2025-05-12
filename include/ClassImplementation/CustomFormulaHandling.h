//
// Created by progamers on 5/6/25.
//
#pragma once
#include "HardCodedVars.h"
#include "FractalClass.cuh"
#include "Macros.h"
#include "kernelCodeStr.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


/// Sets a custom formula for fractal iteration using NVRTC (runtime compilation).
/// This function compiles the provided GLSL-like formula string into CUDA PTX code
/// and loads it as a module for execution. It runs the compilation in a separate thread.
template <typename Derived>
std::shared_future<std::string> FractalBase<Derived>::set_custom_formula(const std::string& formula) {

    /// Prevent starting a new compilation if one is already in progress.
    if(is_compiling) {
        std::promise<std::string> promise;
        promise.set_value("Compilation in progress, please wait.");
        return promise.get_future();
    }
    is_compiling = true;
    std::promise<std::string> p;
    std::future<std::string> f = p.get_future();
    current_compile_future = std::move(f);

    progress_compiling_percentage = 0;
    std::string fractal_name_str;

    if(std::is_same<Derived, fractals::julia>::value){
        fractal_name_str = "julia";
    } else {
        fractal_name_str = "mandelbrot";
    }

    /// Launch compilation in a detached thread to avoid blocking the main application.
    compile_thread = std::thread([this, promise = std::move(p), formula, fractal_name_str]() mutable {
        nvrtcProgram prog = nullptr;

        this->log_buffer = ""; // Clear previous compilation log
        std::string lowered_kernel_name_float_str;
        std::string lowered_kernel_name_double_str;
        std::string lowered_kernel_name_ssaa_str;

        try {
            /// Ensure the context is NVRTC (Driver API) for custom formulas.
            /// This also handles initial NVRTC/Driver API setup if needed.
            if (!this->custom_formula) { // Only switch if not already using custom formula context
                set_context(context_type::NVRTC);
            }
            CU_SAFE_CALL(cuCtxSetCurrent(this->ctx));

            this->progress_compiling_percentage = 20;

            /// Read the content of the custom kernel header file.
            const std::ifstream header_file("custom.cuh");
            if (!header_file.is_open()) {
                log_buffer = "Error: Could not open header file: custom.cuh\n";
                throw std::runtime_error(log_buffer);
            }
            std::stringstream buffer;
            buffer << header_file.rdbuf();
            const std::string header_content = buffer.str();
            const char *header_data = header_content.c_str();
            constexpr char *header_name = "custom.cuh"; // Name used by NVRTC for the included header

            /// Construct the full kernel code by prepending and appending template code
            /// around the user-provided `formula`.
            std::string full_kernel_code_local;
            if (fractal_name_str == "julia") {
                full_kernel_code_local = beginning_julia + formula + ending + julia_predefined + antialiasingCode;
            } else { // mandelbrot
                full_kernel_code_local =
                        beginning_mandelbrot + formula + ending + mandelbrot_predefined + antialiasingCode;
            }

            /// Create an NVRTC program object.
            NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, full_kernel_code_local.c_str(),
                                               ("custom_" + fractal_name_str + ".cu").c_str(), 1, &header_data,
                                               &header_name));

            /// Register kernel names with NVRTC to get their mangled (lowered) names later.
            NVRTC_SAFE_CALL(
                    nvrtcAddNameExpression(prog, ("fractal_rendering_" + fractal_name_str + "<float>").c_str()));
            NVRTC_SAFE_CALL(
                    nvrtcAddNameExpression(prog, ("fractal_rendering_" + fractal_name_str + "<double>").c_str()));
            NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "ANTIALIASING_SSAA4"));

            this->progress_compiling_percentage = 30;

            /// Set NVRTC compilation options (GPU architecture, fast math).
            std::vector<const char *> compile_options;
            compile_options.push_back(compute_capability.c_str());
            compile_options.push_back("--use_fast_math");
            const char **opts = compile_options.data();
            int num_opts = compile_options.size();

            this->progress_compiling_percentage = 35;

            nvrtcResult compileResult = nvrtcCompileProgram(prog, num_opts, opts);

            size_t logSize = 0;
            if (prog) { NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize)); }
            if (logSize > 1 && prog) { // logSize > 1 means there's actual content
                log_buffer.resize(logSize);
                NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log_buffer[0]));
            }

            if (compileResult != NVRTC_SUCCESS) {
                std::cerr << "---------------------\n";
                std::cerr << "NVRTC Compilation Failed for " << fractal_name_str << ":\n";
                std::cerr << "Result Code: " << nvrtcGetErrorString(compileResult) << "\n";
                std::cerr << "---------------------\n";
                std::cerr << "Compilation Log:\n";
                std::cerr << log_buffer << std::endl;
                std::cerr << "---------------------\n\n";
                log_buffer = "Compilation FAILED:\n" + log_buffer;
                // Means it's first time compilation, and we can switch back to CUDA
                if(!module_loaded) set_context(context_type::CUDA);
                throw std::runtime_error("NVRTC Compilation Failed");
            } else {
                std::cout << "NVRTC Compilation Succeeded for " << fractal_name_str << ".\n";
                if (!log_buffer.empty() && log_buffer.length() > 1) {
                    std::cout << "---------------------\n";
                    std::cout << "Compilation Log (Warnings/Info):\n";
                    std::cout << log_buffer << std::endl;
                    std::cout << "---------------------\n\n";
                }
                this->progress_compiling_percentage = 75;
            }

            /// Retrieve the mangled (lowered) names of the compiled kernels.
            const char *lowered_name_float_ptr = nullptr;
            if (prog) {
                NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, ("fractal_rendering_" + fractal_name_str + "<float>").c_str(),
                                                    &lowered_name_float_ptr));
            }
            if (!lowered_name_float_ptr)
                throw std::runtime_error(
                        "Could not get lowered name for fractal_rendering_" + fractal_name_str + "<float>");
            lowered_kernel_name_float_str = lowered_name_float_ptr;

            const char *lowered_name_double_ptr = nullptr;
            if (prog) {
                NVRTC_SAFE_CALL(
                        nvrtcGetLoweredName(prog, ("fractal_rendering_" + fractal_name_str + "<double>").c_str(),
                                            &lowered_name_double_ptr));
            }
            if (!lowered_name_double_ptr)
                throw std::runtime_error(
                        "Could not get lowered name for fractal_rendering_" + fractal_name_str + "<double>");
            lowered_kernel_name_double_str = lowered_name_double_ptr;

            const char *lowered_name_ssaa_ptr = nullptr;
            if (prog) { NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "ANTIALIASING_SSAA4", &lowered_name_ssaa_ptr)); }
            if (!lowered_name_ssaa_ptr) throw std::runtime_error("Could not get lowered name for ANTIALIASING_SSAA4");
            lowered_kernel_name_ssaa_str = lowered_name_ssaa_ptr;

            this->progress_compiling_percentage = 85;

            /// Get the compiled PTX code.
            size_t ptxSize = 0;
            if (prog) { NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize)); }
            std::vector<char> ptx;
            if (ptxSize > 0) {
                ptx.resize(ptxSize);
                NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
            } else if (ptxSize == 0 && compileResult == NVRTC_SUCCESS) {
                std::cerr << "Warning: NVRTC compilation produced 0-byte PTX for " << fractal_name_str << ".\n";
            }

            if (prog) {
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
                prog = nullptr;
            }

            /// Unload any previously loaded NVRTC module and reset function pointers.
            if (this->module_loaded && this->module) { CU_SAFE_CALL(cuModuleUnload(this->module)); }

            this->progress_compiling_percentage = 90;

            if (ptx.empty()) {
                throw std::runtime_error("PTX vector is empty. Cannot load module.");
            }
            /// Load the PTX code into a CUDA module using the Driver API.
            CU_SAFE_CALL(cuModuleLoadDataEx(&this->module, ptx.data(), 0, nullptr, nullptr));
            this->module_loaded = true;
            if (!this->module) {
                throw std::runtime_error("Internal Error: Module is null after successful loading attempt.");
            }

            this->progress_compiling_percentage = 95;

            /// Get function handles for the compiled kernels from the loaded module.
            if (lowered_kernel_name_float_str.empty())
                throw std::runtime_error("Internal Error: Float lowered name string is empty for GetFunction.");
            CU_SAFE_CALL(cuModuleGetFunction(&this->kernelFloat, this->module, lowered_kernel_name_float_str.c_str()));

            if (lowered_kernel_name_double_str.empty())
                throw std::runtime_error("Internal Error: Double lowered name string is empty for GetFunction.");
            CU_SAFE_CALL(cuModuleGetFunction(&this->kernelDouble, this->module,
                                             lowered_kernel_name_double_str.c_str()));

            if (lowered_kernel_name_ssaa_str.empty())
                throw std::runtime_error("Internal Error: SSAA lowered name string is empty for GetFunction.");
            CU_SAFE_CALL(cuModuleGetFunction(&this->kernelAntialiasing, this->module,
                                             lowered_kernel_name_ssaa_str.c_str()));


            this->custom_formula = true; // Mark that a custom formula is now active
            this->progress_compiling_percentage = 100;

            promise.set_value(log_buffer); // Set the future with the compilation log (success or warnings)

        } catch (const std::exception &e) {
            std::cerr << "Caught Exception during NVRTC/CUDA setup for " << fractal_name_str << ": " << e.what()
                      << std::endl;
            if (!log_buffer.empty()) log_buffer += "\n";
            log_buffer += "Exception: ";
            log_buffer += e.what();

            if (prog) { // Ensure NVRTC program is destroyed even on error
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); // This might throw if prog is already bad, but NVRTC_SAFE_CALL handles it
                prog = nullptr;
            }

            this->progress_compiling_percentage = -1; // Indicate error in progress

            promise.set_value(log_buffer); // Set the future with the error log

        } catch (...) { // Catch-all for any other exceptions
            std::cerr << "Caught Unknown Exception during NVRTC/CUDA setup for " << fractal_name_str << std::endl;
            log_buffer = "Caught Unknown Exception during compilation.";

            if (prog) {
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
                prog = nullptr;
            }
            this->progress_compiling_percentage = -1;

            promise.set_value(log_buffer);
        }
        this->is_compiling = false; // Mark compilation as finished
    });
    compile_thread.detach(); // Detach the compilation thread
    return current_compile_future.share(); // Return a shared future for the result
}

/// Sets the CUDA context for fractal rendering (CUDA Runtime API vs. NVRTC/Driver API).
/// This involves freeing existing GPU resources associated with the old context
/// and allocating/initializing resources for the new context.
template <typename Derived>
void FractalBase<Derived>::set_context(context_type contx) {
    if(contx == context) return; // No change if the context is already the same

    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();

    if(contx == context_type::NVRTC) { // Switching TO NVRTC/Driver API
        if(!initialized_nvrtc){ // Global NVRTC/Driver API initialization
            CU_SAFE_CALL(cuInit(0));
            initialized_nvrtc = true;
        }
        CU_SAFE_CALL(cuDeviceGet(&device, 0));
        CU_SAFE_CALL(cuCtxCreate(&ctx, 0, device)); // Create a Driver API context
        CU_SAFE_CALL(cuCtxSetCurrent(ctx));        // Set it as current
        created_context = true;                  // Mark that this instance created the context
    }
    else { // Switching TO CUDA Runtime API
        if (ctx && created_context) { // If this instance created a Driver API context, destroy it
            CU_SAFE_CALL(cuCtxDestroy(ctx));
            ctx = nullptr;
            created_context = false;
        }
        if (device) {
            device = 0; // Reset device handle
        }
        /// For CUDA Runtime API, `cudaSetDevice` is typically sufficient for context management.
        /// The Runtime API implicitly handles context creation/management per thread.
        CUDA_SAFE_CALL(cudaSetDevice(0));
    }
    context = contx; // Update the active context type

    /// Re-allocate resources for the new context.
    ALLOCATE_ALL_IMAGE_MEMORY();
    ALLOCATE_ALL_NON_IMAGE_MEMORY();

    /// Custom formula flag is tied to the NVRTC context.
    if(context == context_type::NVRTC) custom_formula = true;
    else custom_formula = false;
}