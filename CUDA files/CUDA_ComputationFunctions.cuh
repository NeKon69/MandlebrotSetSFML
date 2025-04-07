#pragma once
#include <cuda_runtime.h>
#include <TGUI/Backend/SFML-Graphics.hpp>

// Device functions (inline definitions)
inline __device__ float complex_mult_real(float real1, float imag1, float real2, float imag2) {
    return real1 * real2 - imag1 * imag2;
}

inline __device__ float complex_mult_imag(float real1, float imag1, float real2, float imag2) {
    return 2 * real1 * imag2;
}

inline __device__ float dev_abs(float x) {
    return (x >= 0) ? x : -x;
}

inline __device__ float dev_log2(float x) {
    return log(x) / log(2.0f);
}

inline __device__ float dev_sqrt(float x) {
    return sqrt(x);
}

inline __device__ float complex_abs2(float real, float imag) {
    return real * real + imag * imag;
}

inline __device__ float Gradient(float current_iteration, float max_iter) {
    if (current_iteration >= max_iter) return 0.0;
    // comment following two lines to make the colors less appealing on the screen (less of the colors circle the black thing on the screen)
    current_iteration = dev_sqrt(current_iteration);
    max_iter = dev_sqrt(max_iter);
    return (current_iteration) / static_cast<float>(max_iter);
}

inline __device__ sf::Color getPaletteColor(int index, int paletteSize, sf::Color* palette) {
    index = (index < 0) ? 0 : ((index >= paletteSize) ? paletteSize - 1 : index);
    return palette[index];
}

__global__ void ANTIALIASING_SSAA4(unsigned char* src, unsigned char* dest, int src_width, int src_height, int dest_width, int dest_height);

// Host utility functions (declarations only)
extern void cudaCheckError();
extern void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b);

extern std::vector<sf::Color> CreateBlackOWhitePalette(int numColors);
extern std::vector<sf::Color> createHSVPalette(int numColors, double startHue = 0.0);
extern std::vector<sf::Color> BluePlusBlackWhitePalette(int numColors);

extern std::vector<sf::Color> CreateOscillatingGrayscalePalette(int numColors, int frequency = 5);

extern std::vector<sf::Color> CreateInterpolatedPalette(int numColors);

extern std::vector<sf::Color> CreatePastelPalette(int numColors);

extern std::vector<sf::Color> CreateFirePalette(int numColors);

extern std::vector<sf::Color> CreateWaterPalette(int numColors);

extern std::vector<sf::Color> CreateCyclicHSVPpalette(int numColors, int cycles = 1);

extern std::vector<sf::Color> CreateFractalPatternPalette(int numColors);

extern std::vector<sf::Color> CreatePerlinNoisePalette(int numColors, int seed = 0);

extern std::vector<sf::Color> CreateSunsetPalette(int numColors);

extern std::vector<sf::Color> CreatePsychedelicWavePalette(int numColors, float freqH = 3.0f, float freqS = 1.5f, float freqV = 2.0f);

extern std::vector<sf::Color> CreateIceCavePalette(int numColors);

extern std::vector<sf::Color> CreateAccretionDiskPalette(int numColors, float power = 1.f);

extern std::vector<sf::Color> CreateElectricNebulaPalette(int numColors);

extern std::vector<sf::Color> CreateDeepSpaceWideVeinsPalette(int numColors);
