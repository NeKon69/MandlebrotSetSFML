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
std::optional<std::string> FractalBase<Derived>::set_custom_formula(const std::string formula) {
    INIT_CU_RESOURCES;
    switch(std::is_same<Derived, fractals::julia>::value){
        case 1:
            kernel_code = beginning_julia + formula + ending + julia_predefined + antialiasingCode;
            COMPILE_PROGRAM("julia");
            break;
        case 0:
            kernel_code = beginning_mandelbrot + formula + ending + mandelbrot_predefined + antialiasingCode;
            COMPILE_PROGRAM("mandelbrot");
    }
    return "";
}

template <typename Derived>
/// \brief Set the context for the fractal rendering.
/// This function allows switching between CUDA and NVRTC contexts.
/// It frees existing resources and allocates new ones based on the selected context.
void FractalBase<Derived>::set_context(context_type contx) {
    if(contx == context) return;

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