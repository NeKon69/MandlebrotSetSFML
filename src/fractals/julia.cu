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
	int id = threadIdx.y * blockDim.x + threadIdx.x;

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


		total_iterations[id] = static_cast<unsigned int>(current_iteration);
		__syncthreads();

		for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
			if (id < s) {
				total_iterations[id] += total_iterations[id + s];
			}
			__syncthreads();
		}

		if (id == 0) {
			atomicAdd(d_total_iterations, total_iterations[0]);
		}

		unsigned char r, g, b;
		if (current_iteration == maxIterations) {
			r = g = b = 0; 
		}

		else {
            float smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));

            const float cycle_scale_factor = 25.0f;
            float virtual_pos = smooth_iteration * cycle_scale_factor;

            float normalized_pingpong = fmodf(virtual_pos / static_cast<float>(paletteSize -1), 2.0f);
            if (normalized_pingpong < 0.0f) {
                normalized_pingpong += 2.0f;
            }

            float t_interp;
            if (normalized_pingpong <= 1.0f) {
                t_interp = normalized_pingpong;
            } else {
                t_interp = 2.0f - normalized_pingpong;
            }

            float float_index = t_interp * (paletteSize - 1);

            int index1 = static_cast<int>(floorf(float_index));
            int index2 = min(paletteSize - 1, index1 + 1);

            index1 = max(0, index1);

            float t_local = fmodf(float_index, 1.0f);
            if (t_local < 0.0f) t_local += 1.0f;

            sf::Color color1 = getPaletteColor(index1, paletteSize, d_palette);
            sf::Color color2 = getPaletteColor(index2, paletteSize, d_palette);

            float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
            float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
            float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

            r = static_cast<unsigned char>(max(0.0f, min(255.0f, r_f)));
            g = static_cast<unsigned char>(max(0.0f, min(255.0f, g_f)));
            b = static_cast<unsigned char>(max(0.0f, min(255.0f, b_f)));
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


__global__ void fractal_rendering(
        unsigned char* pixels, size_t size_of_pixels, int width, int height,
        double zoom_x, double zoom_y, double x_offset, double y_offset,
        sf::Color* d_palette, int paletteSize, double maxIterations, unsigned int* d_total_iterations,
        double cReal, double cImaginary) {

    __shared__ unsigned int total_iterations[1024];

    maxIterations = double(maxIterations);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int id = threadIdx.y * blockDim.x + threadIdx.x;

    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }


    size_t expected_size = width * height * 4;

    float scale_factor = (float)size_of_pixels / expected_size;



    if (x < width && y < height) {
        double real = cReal;
        double imag = cImaginary;
        double new_real, new_imag;
        double z_real = x / zoom_x - x_offset;;
        double z_imag = y / zoom_y - y_offset;
        double current_iteration = 0;
        double z_comp = z_real * z_real + z_imag * z_imag;

        while (z_comp < 4 && current_iteration < maxIterations) {
            new_real = (z_real * z_real - z_imag * z_imag) + real;
            z_imag = 2 * z_real * z_imag + imag;
            z_real = new_real;
            z_comp = z_real * z_real + z_imag * z_imag;
            current_iteration++;
        }


        total_iterations[id] = static_cast<unsigned int>(current_iteration);
        __syncthreads();

        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (id < s) {
                total_iterations[id] += total_iterations[id + s];
            }
            __syncthreads();
        }

        if (id == 0) {
            atomicAdd(d_total_iterations, total_iterations[0]);
        }

        unsigned char r, g, b;
        if (current_iteration == maxIterations) {
            r = g = b = 0;
        }

        else {
            float smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));

            const float cycle_scale_factor = 25.0f;
            float virtual_pos = smooth_iteration * cycle_scale_factor;

            float normalized_pingpong = fmodf(virtual_pos / static_cast<float>(paletteSize -1), 2.0f);
            if (normalized_pingpong < 0.0f) {
                normalized_pingpong += 2.0f;
            }

            float t_interp;
            if (normalized_pingpong <= 1.0f) {
                t_interp = normalized_pingpong;
            } else {
                t_interp = 2.0f - normalized_pingpong;
            }

            float float_index = t_interp * (paletteSize - 1);

            int index1 = static_cast<int>(floorf(float_index));
            int index2 = min(paletteSize - 1, index1 + 1);

            index1 = max(0, index1);

            float t_local = fmodf(float_index, 1.0f);
            if (t_local < 0.0f) t_local += 1.0f;

            sf::Color color1 = getPaletteColor(index1, paletteSize, d_palette);
            sf::Color color2 = getPaletteColor(index2, paletteSize, d_palette);

            float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
            float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
            float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

            r = static_cast<unsigned char>(max(0.0f, min(255.0f, r_f)));
            g = static_cast<unsigned char>(max(0.0f, min(255.0f, g_f)));
            b = static_cast<unsigned char>(max(0.0f, min(255.0f, b_f)));
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
