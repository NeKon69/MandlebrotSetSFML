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
template<typename T>
__global__ void fractal_rendering(
        unsigned char* pixels, unsigned long size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        Color* d_palette, int paletteSize, T maxIterations, unsigned int* d_total_iterations)
{


    const unsigned int x =   blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y =   blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;

    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }

    const size_t expected_size = width * height * 4;

    const T scale_factor = static_cast<T>(size_of_pixels) / static_cast<T>(expected_size);

    if (x < width && y < height) {
        __shared__ unsigned int total_iterations[1024];
        const T real = x / zoom_x - x_offset;
        const T imag = y / zoom_y - y_offset;
        T new_real = 0.0;
        T z_real = 0.0;
        T z_imag = 0.0;
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
//        __syncthreads();

        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (id < s) {
                total_iterations[id] += total_iterations[id + s];
            }
//            __syncthreads();
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
            T smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));

            const T cycle_scale_factor = 25.0f;
            T virtual_pos = smooth_iteration * cycle_scale_factor;
            
            T normalized_pingpong = fmodf(virtual_pos / static_cast<T>(paletteSize -1), 2.0f);
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
            if (t_local < 0.0f) t_local += 1.0f;
            
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
        unsigned char*, size_t, unsigned int, unsigned int,
        float, float, float, float,
        Color*, int, float, unsigned int*);

template __global__ void fractal_rendering<double>(
        unsigned char*, size_t, unsigned int, unsigned int,
        double, double, double, double,
        Color*, int, double, unsigned int*);


//__global__ void fractal_rendering(
//    unsigned char* pixels, const size_t size_of_pixels, const int width, const int height,
//    const float zoom_x, const float zoom_y, const float x_offset, const float y_offset,
//    sf::Color* d_palette, const int paletteSize, const float maxIterations, unsigned int* d_total_iterations) {
//
//    __shared__ unsigned int total_iterations[1024];
//
//    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;
//
//    if (x == 0 && y == 0) {
//        *d_total_iterations = 0;
//    }
//    __syncthreads();
//
//    const size_t expected_size = static_cast<size_t>(width) * height * 4;
//    const float scale_factor = static_cast<float>(size_of_pixels) / static_cast<float>(expected_size);
//
//    if (x < width && y < height) {
//
//        const float real = static_cast<float>(x) / zoom_x - x_offset;
//        const float imag = static_cast<float>(y) / zoom_y - y_offset;
//        float z_real = 0.0f, z_imag = 0.0f;
//        float current_iteration = 0.0f;
//        float z_comp = z_real * z_real + z_imag * z_imag;
//
//        while (z_comp < 4.0f && current_iteration < maxIterations) {
//            const float new_real = (z_real * z_real - z_imag * z_imag) + real;
//            z_imag = 2.0f * z_real * z_imag + imag;
//            z_real = new_real;
//            current_iteration++;
//            z_comp = z_real * z_real + z_imag * z_imag;
//        }
//
//        total_iterations[id] = static_cast<unsigned int>(current_iteration);
//        __syncthreads();
//
//        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
//            if (id < s) {
//                total_iterations[id] += total_iterations[id + s];
//            }
//            __syncthreads();
//        }
//
//        if (id == 0) {
//            atomicAdd(d_total_iterations, total_iterations[0]);
//        }
//
//        unsigned char r, g, b;
//        if (current_iteration == maxIterations) {
//            r = g = b = 0;
//        } else {
//            float smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));
//
//            const float cycle_scale_factor = 25.0f;
//            float virtual_pos = smooth_iteration * cycle_scale_factor;
//
//            float normalized_pingpong = fmodf(virtual_pos / static_cast<float>(paletteSize -1), 2.0f);
//            if (normalized_pingpong < 0.0f) {
//                normalized_pingpong += 2.0f;
//            }
//
//            float t_interp;
//            if (normalized_pingpong <= 1.0f) {
//                t_interp = normalized_pingpong;
//            } else {
//                t_interp = 2.0f - normalized_pingpong;
//            }
//
//            float float_index = t_interp * (paletteSize - 1);
//
//            int index1 = static_cast<int>(floorf(float_index));
//            int index2 = min(paletteSize - 1, index1 + 1);
//
//            index1 = max(0, index1);
//
//            float t_local = fmodf(float_index, 1.0f);
//            if (t_local < 0.0f) t_local += 1.0f;
//
//            sf::Color color1 = getPaletteColor(index1, paletteSize, d_palette);
//            sf::Color color2 = getPaletteColor(index2, paletteSize, d_palette);
//
//            float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
//            float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
//            float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));
//
//            r = static_cast<unsigned char>(max(0.0f, min(255.0f, r_f)));
//            g = static_cast<unsigned char>(max(0.0f, min(255.0f, g_f)));
//            b = static_cast<unsigned char>(max(0.0f, min(255.0f, b_f)));
//        }
//
//        const unsigned int base_index = (y * width + x) * 4;
//        for (unsigned int i = 0; i < static_cast<int>(scale_factor * 4); i += 4) {
//            const unsigned int index = base_index + i;
//            pixels[index] = r;
//            pixels[index + 1] = g;
//            pixels[index + 2] = b;
//            pixels[index + 3] = 255;
//        }
//    }
//}