#include "julia.cuh"
#include <iostream>

template<typename T>
__global__ void fractal_rendering(
        unsigned char* pixels, unsigned long size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        Color* d_palette, int paletteSize, T maxIterations, unsigned int* d_total_iterations,
        T cReal, T cImaginary)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;

    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }

    const size_t expected_size = width * height * 4;
    const T scale_factor = static_cast<T>(size_of_pixels) / static_cast<T>(expected_size);

    if (x < width && y < height) {
        __shared__ unsigned int total_iterations[1024];
        T z_real = x / zoom_x - x_offset;
        T z_imag = y / zoom_y - y_offset;
        T real = cReal;
        T imag = cImaginary;

        T new_real = 0.0;
        T current_iteration = 0;
        T z_comp = z_real * z_real + z_imag * z_imag;

        while (z_comp < 4 && current_iteration < maxIterations) {
            new_real = (z_real * z_real - z_imag * z_imag) + real;
            z_imag = 2 * z_real * z_imag + imag;
            z_real = new_real;
            z_comp = z_real * z_real + z_imag * z_imag;
            current_iteration++;
        }

        total_iterations[id] = static_cast<unsigned int>(current_iteration);

        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (id < s) {
                total_iterations[id] += total_iterations[id + s];
            }
        }

        if (id == 0) {
            atomicAdd(d_total_iterations, total_iterations[0]);
        }

        unsigned char r, g, b;
        if (current_iteration == maxIterations) {
            r = g = b = 0;
        } else {
            T smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));
            const T cycle_scale_factor = 25.0f;
            T virtual_pos = smooth_iteration * cycle_scale_factor;

            T normalized_pingpong = fmodf(virtual_pos / static_cast<T>(paletteSize - 1), 2.0f);
            if (normalized_pingpong < 0.0f) {
                normalized_pingpong += 2.0f;
            }

            T t_interp;
            if (normalized_pingpong <= 1.0f) {
                t_interp = normalized_pingpong;
            } else {
                t_interp = 2.0f - normalized_pingpong;
            }

            T float_index = t_interp * (paletteSize - 1);
            int index1 = static_cast<int>(floorf(float_index));
            int index2 = min(paletteSize - 1, index1 + 1);
            index1 = max(0, index1);

            T t_local = fmodf(float_index, 1.0f);
            if (t_local < 0.0f) {
                t_local += 1.0f;
            }

            Color color1 = getPaletteColor(index1, paletteSize, d_palette);
            Color color2 = getPaletteColor(index2, paletteSize, d_palette);

            float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
            float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
            float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

            r = static_cast<unsigned char>(max(0.0f, min(255.0f, r_f)));
            g = static_cast<unsigned char>(max(0.0f, min(255.0f, g_f)));
            b = static_cast<unsigned char>(max(0.0f, min(255.0f, b_f)));
        }

        const unsigned int base_index = (y * width + x) * 4;
        for (int i = 0; i < scale_factor * 4; i += 4) {
            const unsigned int index = base_index + i;
            pixels[index] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
        }
    }
}

template __global__ void fractal_rendering<float>(
        unsigned char*, unsigned long, unsigned int, unsigned int,
        float, float, float, float,
        Color*, int, float, unsigned int*,
        float, float);

template __global__ void fractal_rendering<double>(
        unsigned char*, unsigned long, unsigned int, unsigned int,
        double, double, double, double,
        Color*, int, double, unsigned int*,
        double, double);