//
// Created by progamers on 5/6/25.
//
#pragma once
#include "FractalClass.cuh"
#include "Macros.h"
#include "HardCodedVars.h"
#include <iostream>

/// Finalizes the rendering process.
/// Transfers pixel data from GPU to CPU, applies anti-aliasing if enabled,
/// updates the SFML texture, and calculates the fractal "hardness".
template <typename Derived>
void FractalBase<Derived>::post_processing() {
    if(!isCudaAvailable){
        image.resize({ basic_width, basic_height }, pixels);
        hardness_coeff = *h_total_iterations / (width * height * 1.0);
    }
    else if (antialiasing) {
        // SSAA rendering
        set_grid(BLOCK_SIZE_ANTIALIASING);
        if(!custom_formula){
            ANTIALIASING_SSAA4<<<dimGrid, dimBlock, 0, stream>>>(d_pixels, ssaa_buffer, basic_width * 2, basic_height * 2, basic_width, basic_height);
            cudaMemcpyAsync(compressed, ssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        else {
            CU_SAFE_CALL(cuCtxSetCurrent(ctx));
            void* params[] = { &cu_d_pixels, &CUssaa_buffer, &width, &height, &basic_width, &basic_height };
            CU_SAFE_CALL(cuLaunchKernel(kernelAntialiasing,
                                        dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0,
                                        CUss,
                                        params,
                                        nullptr
            ));
            CU_SAFE_CALL(cuMemcpyDtoHAsync((void**)compressed, CUssaa_buffer, basic_height * basic_width * 4 * sizeof(unsigned char),CUss));
            CU_SAFE_CALL(cuStreamSynchronize(CUss));
        }

        image.resize({ basic_width, basic_height }, compressed);

    }
    else {
        if(custom_formula) {
            CU_SAFE_CALL(cuCtxSetCurrent(ctx));
            CU_SAFE_CALL(cuMemcpyDtoHAsync(pixels, cu_d_pixels, basic_width * basic_height * 4 * sizeof(unsigned char), CUss));
            /// Copy the total iteration count from device to host for hardness calculation.
            CU_SAFE_CALL(cuMemcpyDtoHAsync(h_total_iterations, cu_d_total_iterations, sizeof(int), CUssData));
            CU_SAFE_CALL(cuStreamSynchronize(CUss));
        }
        else {
            cudaMemcpyAsync(pixels, d_pixels, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
            /// Copy the total iteration count from device to host for hardness calculation.
            cudaMemcpyAsync(h_total_iterations, d_total_iterations, sizeof(int), cudaMemcpyDeviceToHost, dataStream);
            // THERE AIN'T NO WAY I AM DOING THAT SYNC LOGIC, JUST LET IT BE
            cudaStreamSynchronize(stream);
        }
        /// Calculate the "hardness coefficient" as the average number of iterations per pixel.
        hardness_coeff = *h_total_iterations / (width * height * 1.0);
        image.resize({ basic_width, basic_height }, pixels);
    }
    texture.resize({basic_width, basic_height}, true);
    /// Load the pixel data into the SFML texture. The 'true' argument sets sRgb for pixels color.
    if(!texture.loadFromImage(image, true)) {
        std::cerr << "Data corrupted!\n";
    }
    sprite.setTexture(texture, true);
    sprite.setPosition({ 0, 0 });
}

/// Draws the fractal sprite and the iteration path onto the render target.
template <typename Derived>
void FractalBase<Derived>::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    states.transform *= getTransform();
    target.draw(sprite, states);
    /// Draw the iteration path line strip. `drawen_iterations` controls how many points of the path are drawn.
    target.draw(iterationline, 0, drawen_iterations, states);
}