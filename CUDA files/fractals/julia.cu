#include "julia.cuh"
#include <iostream>

__global__ void fractal_rendering(
	unsigned char* pixels, size_t size_of_pixels, int width, int height,
	float zoom_x, float zoom_y, float x_offset, float y_offset,
	sf::Color* d_palette, int paletteSize, float maxIterations, bool* stopFlagDevice,
	float cReal, float cImaginary) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	size_t expected_size = width * height * 4;

	float scale_factor = (float)size_of_pixels / expected_size;



	if (x < width && y < height) {
		float real = cReal;
		float imag = cImaginary;
		float new_real, new_imag;
		float z_real = x / zoom_x - x_offset;;
		float z_imag = y / zoom_y - y_offset;
		float current_iteration = 0;

		while (complex_abs2(z_real, z_imag) < 4 && current_iteration < maxIterations) {
			new_real = complex_mult_real(z_real, z_imag, z_real, z_imag) + real;
			new_imag = complex_mult_imag(z_real, z_imag, z_real, z_imag) + imag;
			z_real = new_real;
			z_imag = new_imag;
			current_iteration++;

		}

		unsigned char r, g, b;
		if (current_iteration == maxIterations) {
			r = g = b = 0; 
		}

		else {
			// Smooth iteration count
			current_iteration = current_iteration + 1 - dev_log2(dev_log2(dev_abs(dev_sqrt(complex_abs2(z_real, z_imag)))));
			// Calculate gradient value
			float gradient = Gradient(current_iteration, maxIterations);
			// Map gradient to palette index
			int index = static_cast<int>(gradient * (paletteSize - 1));
			sf::Color color = getPaletteColor(index, paletteSize, d_palette);
			r = color.r;
			g = color.g;
			b = color.b;
		}

		int base_index = (y * width + x) * 4;
		for (int i = 0; i < scale_factor * 4; i += 4) {
			int index = base_index + i;
			pixels[index] = r;
			pixels[index + 1] = g;
			pixels[index + 2] = b;
			pixels[index + 3] = 255;
		}
	}

}
