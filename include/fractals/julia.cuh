#pragma once
#include "../CUDA_ComputationFunctions.cuh"

template <typename T>
__global__ void fractal_rendering(
        unsigned char* pixels, size_t size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        sf::Color* d_palette, int paletteSize, T maxIterations, unsigned int* d_total_iterations,
	    T cReal, T cImaginary
);