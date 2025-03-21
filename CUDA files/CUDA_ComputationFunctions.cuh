#pragma once
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

// Device functions (inline definitions)
inline __device__ double complex_mult_real(double real1, double imag1, double real2, double imag2) {
    return real1 * real2 - imag1 * imag2;
}

inline __device__ double complex_mult_imag(double real1, double imag1, double real2, double imag2) {
    return real1 * imag2 + imag1 * real2;
}

inline __device__ double dev_abs(double x) {
    return (x >= 0) ? x : -x;
}

inline __device__ double dev_log2(double x) {
    return log(x) / log(2.0);
}

inline __device__ double dev_sqrt(double x) {
    return sqrt(x);
}

inline __device__ double complex_abs2(double real, double imag) {
    return real * real + imag * imag;
}

inline __device__ double Gradient(double current_iteration, double max_iter) {
    if (current_iteration >= max_iter) return 0.0;
    // comment following two lines to make the colors less appealing on the screen (less of the colors circle the black thing on the screen)
    current_iteration = dev_sqrt(current_iteration);
    max_iter = dev_sqrt(max_iter);
    return (current_iteration) / static_cast<double>(max_iter);
}

inline __device__ sf::Color getPaletteColor(int index, int paletteSize, sf::Color* palette) {
    index = (index < 0) ? 0 : ((index >= paletteSize) ? paletteSize - 1 : index);
    return palette[index];
}

__global__ void ANTIALIASING_SSAA4(unsigned char* src, unsigned char* dest, int src_width, int src_height, int dest_width, int dest_height);

// Host utility functions (declarations only)
extern void cudaCheckError();
extern void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b);
extern std::vector<sf::Color> createHSVPalette(int numColors);

extern std::vector<sf::Color> palette;