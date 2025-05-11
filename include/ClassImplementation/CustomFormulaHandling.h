//
// Created by progamers on 5/6/25.
//
#pragma once
#include "ClassImplementation/FractalClass.cuh"
#include "ClassImplementation/Macros.h"
#include "kernelCodeStr.h"
#include <iostream>
#include <fstream>
#include <string>



template <typename Derived>
/// Formula should be like this one\n
/// new_real = z_real * z_real - z_imag * z_imag + real;\n
/// z_imag =  2 * z_real * z_imag + imag;
std::shared_future<std::string> FractalBase<Derived>::set_custom_formula(const std::string& formula) {

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

    compile_thread = std::thread([this, promise = std::move(p), formula, fractal_name_str]() mutable {
        nvrtcProgram prog = nullptr;

        this->log_buffer = "";
        std::string lowered_kernel_name_float_str;
        std::string lowered_kernel_name_double_str;
        std::string lowered_kernel_name_ssaa_str;

        try {


            if (!this->custom_formula) {
                set_context(context_type::NVRTC);
            }
            CU_SAFE_CALL(cuCtxSetCurrent(this->ctx));

            this->progress_compiling_percentage = 20;

            // Read header file
            const std::ifstream header_file("../include/fractals/custom.cuh");
            if (!header_file.is_open()) {
                log_buffer = "Error: Could not open header file: ../include/fractals/custom.cuh\n";
                throw std::runtime_error(log_buffer);
            }
            std::stringstream buffer;
            buffer << header_file.rdbuf();
            const std::string header_content = buffer.str();
            const char *header_data = header_content.c_str();
            constexpr char *header_name = "custom.cuh";

            // Form full kernel code locally
            std::string full_kernel_code_local;
            if (fractal_name_str == "julia") {
                // Assuming these are members accessible via this->
                full_kernel_code_local = beginning_julia + formula + ending + julia_predefined + antialiasingCode;
            } else {
                full_kernel_code_local =
                        beginning_mandelbrot + formula + ending + mandelbrot_predefined + antialiasingCode;
            }

            // Create program
            // Use local prog and full_kernel_code_local, captured fractal_name_str
            NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, full_kernel_code_local.c_str(),
                                               ("custom_" + fractal_name_str + ".cu").c_str(), 1, &header_data,
                                               &header_name));

            // Add name expressions
            // Use local prog, captured fractal_name_str
            NVRTC_SAFE_CALL(
                    nvrtcAddNameExpression(prog, ("fractal_rendering_" + fractal_name_str + "<float>").c_str()));
            NVRTC_SAFE_CALL(
                    nvrtcAddNameExpression(prog, ("fractal_rendering_" + fractal_name_str + "<double>").c_str()));
            NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "ANTIALIASING_SSAA4"));

            this->progress_compiling_percentage = 30;

            // Prepare compile options
            std::vector<const char *> compile_options;
            // Use this->gpu_architecture_option (member)
            compile_options.push_back(compute_capability.c_str());
            compile_options.push_back("--use_fast_math");
            const char **opts = compile_options.data();
            int num_opts = compile_options.size();

            this->progress_compiling_percentage = 35;

            // Compile the program
            nvrtcResult compileResult = nvrtcCompileProgram(prog, num_opts, opts);

            // Get compilation log
            size_t logSize = 0;
            if (prog) { NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize)); }
            if (logSize > 1 && prog) {
                log_buffer.resize(logSize);
                NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log_buffer[0]));
            }

            if (compileResult != NVRTC_SUCCESS) {
                // Error handling, print log to cerr
                std::cerr << "---------------------\n";
                std::cerr << "NVRTC Compilation Failed for " << fractal_name_str << ":\n";
                std::cerr << "Result Code: " << nvrtcGetErrorString(compileResult) << "\n";
                std::cerr << "---------------------\n";
                std::cerr << "Compilation Log:\n";
                std::cerr << log_buffer << std::endl;
                std::cerr << "---------------------\n\n";
                log_buffer = "Compilation FAILED:\n" + log_buffer; // Prepend error message to log
                throw std::runtime_error("NVRTC Compilation Failed"); // Go to catch block
            } else {
                // Success, print log if warnings to cout
                std::cout << "NVRTC Compilation Succeeded for " << fractal_name_str << ".\n";
                if (!log_buffer.empty() && log_buffer.length() > 1) {
                    std::cout << "---------------------\n";
                    std::cout << "Compilation Log (Warnings/Info):\n";
                    std::cout << log_buffer << std::endl;
                    std::cout << "---------------------\n\n";
                }
                this->progress_compiling_percentage = 75;
            }

            // Get lowered kernel names
            // Use local prog, write to this->lowered_kernel_name_..._str (members)
            const char *lowered_name_float_ptr = nullptr;
            if (prog) {
                NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, ("fractal_rendering_" + fractal_name_str + "<float>").c_str(),
                                                    &lowered_name_float_ptr));
            }
            if (!lowered_name_float_ptr)
                throw std::runtime_error(
                        "Could not get lowered name for fractal_rendering_" + fractal_name_str + "<float>");
            lowered_kernel_name_float_str = lowered_name_float_ptr; // Write to member

            const char *lowered_name_double_ptr = nullptr;
            if (prog) {
                NVRTC_SAFE_CALL(
                        nvrtcGetLoweredName(prog, ("fractal_rendering_" + fractal_name_str + "<double>").c_str(),
                                            &lowered_name_double_ptr));
            }
            if (!lowered_name_double_ptr)
                throw std::runtime_error(
                        "Could not get lowered name for fractal_rendering_" + fractal_name_str + "<double>");
            lowered_kernel_name_double_str = lowered_name_double_ptr; // Write to member

            const char *lowered_name_ssaa_ptr = nullptr;
            if (prog) { NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "ANTIALIASING_SSAA4", &lowered_name_ssaa_ptr)); }
            if (!lowered_name_ssaa_ptr) throw std::runtime_error("Could not get lowered name for ANTIALIASING_SSAA4");
            lowered_kernel_name_ssaa_str = lowered_name_ssaa_ptr; // Write to member

            this->progress_compiling_percentage = 85;

            // Get PTX
            size_t ptxSize = 0;
            if (prog) { NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize)); }
            std::vector<char> ptx; // Local variable
            if (ptxSize > 0) {
                ptx.resize(ptxSize);
                NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
            } else if (ptxSize == 0 && compileResult == NVRTC_SUCCESS) {
                std::cerr << "Warning: NVRTC compilation produced 0-byte PTX for " << fractal_name_str << ".\n";
                // Could add some warnings and stuff
            } else if (ptxSize == 0 && compileResult != NVRTC_SUCCESS) {
                // Could add some warnings and stuff
            }


            // Destroy NVRTC program
            if (prog) {
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
                prog = nullptr;
            }

            if (this->module_loaded && this->module) { CU_SAFE_CALL(cuModuleUnload(this->module)); }
            this->module = nullptr;
            this->module_loaded = false;
            this->kernelFloat = nullptr;
            this->kernelDouble = nullptr;
            this->kernelAntialiasing = nullptr;

            this->progress_compiling_percentage = 90;

            if (ptx.empty()) {
                throw std::runtime_error("PTX vector is empty. Cannot load module.");
            }
            CU_SAFE_CALL(cuModuleLoadDataEx(&this->module, ptx.data(), 0, nullptr, nullptr));
            this->module_loaded = true;
            if (!this->module) {
                throw std::runtime_error("Internal Error: Module is null after successful loading attempt.");
            }

            this->progress_compiling_percentage = 95;

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


            this->custom_formula = true;
            this->progress_compiling_percentage = 100;

            promise.set_value(log_buffer);

        } catch (const std::exception &e) {
            std::cerr << "Caught Exception during NVRTC/CUDA setup for " << fractal_name_str << ": " << e.what()
                      << std::endl;
            if (!log_buffer.empty()) log_buffer += "\n";
            log_buffer += "Exception: ";
            log_buffer += e.what();

            if (prog) {
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
                prog = nullptr;
            }

            if (this->module_loaded && this->module) {
                CUresult unload_res = cuModuleUnload(this->module);
                if (unload_res != CUDA_SUCCESS) {
                    std::cerr << "Error during module unload in error handler: " << unload_res << std::endl;
                }
            }
            this->module = nullptr;
            this->module_loaded = false;
            this->kernelFloat = nullptr;
            this->kernelDouble = nullptr;
            this->kernelAntialiasing = nullptr;
            lowered_kernel_name_float_str.clear();
            lowered_kernel_name_double_str.clear();
            lowered_kernel_name_ssaa_str.clear();
            this->custom_formula = false;

            this->progress_compiling_percentage = -1;

            promise.set_value(log_buffer);

        } catch (...) {
            std::cerr << "Caught Unknown Exception during NVRTC/CUDA setup for " << fractal_name_str << std::endl;
            log_buffer = "Caught Unknown Exception during compilation.";

            if (prog) {
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
                prog = nullptr;
            }
            if (this->module_loaded && this->module) {
                CUresult unload_res = cuModuleUnload(this->module);
                if (unload_res != CUDA_SUCCESS) {
                    std::cerr << "Error during module unload in unknown error handler: " << unload_res << std::endl;
                }
            }
            this->module = nullptr;
            this->module_loaded = false;
            this->kernelFloat = nullptr;
            this->kernelDouble = nullptr;
            this->kernelAntialiasing = nullptr;
            lowered_kernel_name_float_str.clear();
            lowered_kernel_name_double_str.clear();
            lowered_kernel_name_ssaa_str.clear();
            this->custom_formula = false;

            this->progress_compiling_percentage = -1;

            promise.set_value(log_buffer);
        }
        this->is_compiling = false;
    });
    compile_thread.detach();
    return current_compile_future.share();
}

template <typename Derived>
/// \brief Set the context for the fractal rendering.
/// This function allows switching between CUDA and NVRTC contexts.
/// It frees existing resources and allocates new ones based on the selected context.
void FractalBase<Derived>::set_context(context_type contx) {
    if(contx == context) return;
    std::cout << "Switching context of fractal(" << (std::is_same<Derived, fractals::julia>::value? "Julia" : "Mandelbrot") << ")from " << (context == context_type::CUDA ? "CUDA" : "NVRTC") << " to " << (contx == context_type::CUDA ? "CUDA" : "NVRTC") << std::endl;

    // Free existing resources
    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();

    if(context == context_type::CUDA) {
        if(!initialized_nvrtc){
            CU_SAFE_CALL(cuInit(0));
        }
        initialized_nvrtc = true;
        CU_SAFE_CALL(cuDeviceGet(&device, 0));
        CU_SAFE_CALL(cuCtxCreate(&ctx, 0, device));
        CU_SAFE_CALL(cuCtxSetCurrent(ctx));
        created_context = true;
    }
    else {
        if (ctx) {
            created_context = false;
            CU_SAFE_CALL(cuCtxDestroy(ctx));
            ctx = nullptr;
        }
        if (device) {
            device = 0;
        }
        CUDA_SAFE_CALL(cudaSetDevice(0));
    }
    context = contx;

    ALLOCATE_ALL_IMAGE_MEMORY();
    ALLOCATE_ALL_NON_IMAGE_MEMORY();

    if(context == context_type::NVRTC) custom_formula = true;
    else custom_formula = false;
}