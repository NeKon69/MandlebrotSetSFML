#pragma once

__device__ float PI = 3.14159265358979323846f;

#ifdef __cplusplus
struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
    __host__ __device__ Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255) : r(r), g(g), b(b), a(a) {}
    __host__ __device__ Color() : r(0), g(0), b(0), a(255) {}
};
#endif

inline __device__ __host__ double Gradient(double current_iteration, double max_iter) {
    if (current_iteration >= max_iter) return 0.0;
    // comment following two lines to make the colors less appealing on the screen (less of the colors circle the black thing on the screen)
    // current_iteration = sqrt(current_iteration);
    // max_iter = sqrt(max_iter);
    return (current_iteration) / static_cast<double>(max_iter);
}

inline __device__ __host__ Color getPaletteColor(int index, int paletteSize, Color* palette) {
    index = (index < 0) ? 0 : ((index >= paletteSize) ? paletteSize - 1 : index);
    return palette[index];
}

template <typename T>
extern __global__ void fractal_rendering_mandelbrot(
        unsigned char* pixels, size_t size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        Color* d_palette, unsigned int paletteSize, T maxIterations, unsigned int* d_total_iterations
);

template <typename T>
extern __global__ void fractal_rendering_julia(
        unsigned char* pixels, size_t size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        Color* d_palette, unsigned int paletteSize, T maxIterations, unsigned int* d_total_iterations, T cReal, T cImaginary);