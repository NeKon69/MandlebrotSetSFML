#include "mandelbrot.cuh"

/**
 * @brief CUDA kernel function to calculate and render the Mandelbrot set.
 * This kernel is executed in parallel by multiple threads on the GPU.
 * Each thread calculates the color of a single pixel based on its position
 * and the Mandelbrot set algorithm.
 *
 * @param pixels Pointer to the pixel data buffer in device memory.
 * @param size_of_pixels Size of the pixel data buffer.
 * @param width Image width.
 * @param height Image height.
 * @param zoom_x Zoom level along the x-axis.
 * @param zoom_y Zoom level along the y-axis.
 * @param x_offset Offset in the x-direction to move the view.
 * @param y_offset Offset in the y-direction to move the view.
 * @param d_palette Color palette in device memory to color the Mandelbrot set.
 * @param paletteSize Size of the color palette.
 * @param maxIterations Maximum iterations for Mandelbrot calculation.
 * @param d_total_iterations Pointer to store the total number of iterations performed.
 */
__global__ void fractal_rendering(
    unsigned char* pixels, const size_t size_of_pixels, const int width, const int height,
    const double zoom_x, const double zoom_y, const double x_offset, const double y_offset,
	sf::Color* d_palette, const int paletteSize, const double maxIterations, unsigned int* d_total_iterations) {


    const unsigned int x =   blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y =   blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;

    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }

    const size_t expected_size = width * height * 4;

    const double scale_factor = static_cast<double>(size_of_pixels) / static_cast<double>(expected_size);

    if (x < width && y < height) {
        __shared__ unsigned int total_iterations[1024];
        const double real = x / zoom_x - x_offset;
        const double imag = y / zoom_y - y_offset;
        double new_real = 0.0;
        double z_real = 0.0;
        double z_imag = 0.0;
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
            //d_total_iterations += total_iterations[0];
            atomicAdd(d_total_iterations, total_iterations[0]);
        }

        unsigned char r, g, b;
        if (current_iteration == maxIterations) {
            r = g = b = 0;
        }

        else {

            //modulus = hypot(z_real, z_imag);
            //double escape_radius = 2.0;
            //if (modulus > escape_radius) {
            //    double nu = log2(log2(modulus) - log2(escape_radius));
            //    current_iteration = current_iteration + 1 - nu;
            //}
            // Smooth iteration count
            current_iteration = current_iteration + 1 - log2(log2(abs(sqrt(z_real * z_real + z_imag * z_imag))));
            // Calculate gradient value
            const auto gradient = static_cast<float>(Gradient(current_iteration, maxIterations));
            // Map gradient to palette index
            const int index = static_cast<int>(gradient * static_cast<float>(paletteSize - 1));
            const sf::Color color = getPaletteColor(index, paletteSize, d_palette);
            r = color.r;
            g = color.g;
            b = color.b;
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


__global__ void fractal_rendering(
    unsigned char* pixels, const size_t size_of_pixels, const int width, const int height,
    const float zoom_x, const float zoom_y, const float x_offset, const float y_offset,
    sf::Color* d_palette, const int paletteSize, const float maxIterations, unsigned int* d_total_iterations) {

    __shared__ unsigned int total_iterations[1024];

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;

    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }
    __syncthreads();

    const size_t expected_size = static_cast<size_t>(width) * height * 4;
    const float scale_factor = static_cast<float>(size_of_pixels) / static_cast<float>(expected_size);

    if (x < width && y < height) {

        const float real = static_cast<float>(x) / zoom_x - x_offset;
        const float imag = static_cast<float>(y) / zoom_y - y_offset;
        float z_real = 0.0f, z_imag = 0.0f;
        float current_iteration = 0.0f;
        float z_comp = z_real * z_real + z_imag * z_imag;

        while (z_comp < 4.0f && current_iteration < maxIterations) {
            const float new_real = (z_real * z_real - z_imag * z_imag) + real;
            z_imag = 2.0f * z_real * z_imag + imag;
            z_real = new_real;
            current_iteration++;
            z_comp = z_real * z_real + z_imag * z_imag;
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
        } else {
            current_iteration = current_iteration + 1.0f - log2(log2(sqrt(z_real * z_real + z_imag * z_imag)));
            const auto gradient = static_cast<float>(Gradient(current_iteration, maxIterations));
            const int index = static_cast<int>(gradient * static_cast<float>(paletteSize - 1));
            const sf::Color color = getPaletteColor(index, paletteSize, d_palette);
            r = color.r;
            g = color.g;
            b = color.b;
        }

        const unsigned int base_index = (y * width + x) * 4;
        for (unsigned int i = 0; i < static_cast<int>(scale_factor * 4); i += 4) {
            const unsigned int index = base_index + i;
            pixels[index] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
        }
    }
}