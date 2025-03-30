#pragma once
#include "../CUDA_ComputationFunctions.cuh"

__global__ void fractal_rendering(
	unsigned char* pixels, size_t size_of_pixels, int width, int height,
	float zoom_x, float zoom_y, float x_offset, float y_offset,
	sf::Color* d_palette, int paletteSize, float maxIterations, bool* stopFlagDevice,
	float cReal, float cImaginary
);