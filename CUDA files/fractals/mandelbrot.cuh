#pragma once
#include "../CUDA_ComputationFunctions.cuh"

__global__ void fractal_rendering(
    unsigned char* pixels, size_t size_of_pixels, int width, int height,
    double zoom_x, double zoom_y, double x_offset, double y_offset,
    sf::Color* d_palette, int paletteSize, double maxIterations, bool* stopFlagDevice
);