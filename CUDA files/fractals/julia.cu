#include "julia.cuh"
#include <iostream>

__global__ void fractal_rendering(
	unsigned char* pixels, size_t size_of_pixels, int width, int height,
	float zoom_x, float zoom_y, float x_offset, float y_offset,
	sf::Color* d_palette, int paletteSize, float maxIterations, unsigned int* d_total_iterations,
	float cReal, float cImaginary) {

	__shared__ unsigned int total_iterations[1024];

	maxIterations = float(maxIterations);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = threadIdx.y * blockDim.y + threadIdx.x;

	if (x == 0 && y == 0) {
		*d_total_iterations = 0;
	}


	size_t expected_size = width * height * 4;

	float scale_factor = (float)size_of_pixels / expected_size;



	if (x < width && y < height) {
		float real = cReal;
		float imag = cImaginary;
		float new_real, new_imag;
		float z_real = x / zoom_x - x_offset;;
		float z_imag = y / zoom_y - y_offset;
		float current_iteration = 0;
		float z_comp = z_real * z_real + z_imag * z_imag;

		while (z_comp < 4 && current_iteration < maxIterations) {
			new_real = (z_real * z_real - z_imag * z_imag) + real;
			z_imag = 2 * z_real * z_imag + imag;
			z_real = new_real;
			z_comp = z_real * z_real + z_imag * z_imag;
			current_iteration++;
		}


		total_iterations[id] = current_iteration;
		//__syncthreads();

		for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
			if (id < s) {
				total_iterations[id] += total_iterations[id + s];
			}
			//__syncthreads();
		}
		if (id == 0) {
			//d_total_iterations += total_iterations[0];
			atomicAdd(d_total_iterations, total_iterations[0]);
		}

		unsigned char r, g, b;
		if (current_iteration == maxIterations) {
			r = g = b = 0; 
		}

		else {
			current_iteration = current_iteration + 1 - log2(log2(abs(sqrt(z_real * z_real + z_imag * z_imag))));
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
