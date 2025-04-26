#pragma once
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <cuda_runtime.h>

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

__global__ void ANTIALIASING_SSAA4(unsigned char* src, unsigned char* dest, int src_width, int src_height, int dest_width, int dest_height);

// Host utility functions (declarations only)
extern void cudaCheckError();
extern void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b);

extern std::vector<Color> CreateBlackOWhitePalette(int numColors);

extern std::vector<Color> createHSVPalette(int numColors, double startHue = 0.0);

extern std::vector<Color> BluePlusBlackWhitePalette(int numColors);

extern std::vector<Color> CreateOscillatingGrayscalePalette(int numColors, int frequency = 5);

extern std::vector<Color> CreateInterpolatedPalette(int numColors);

extern std::vector<Color> CreatePastelPalette(int numColors);

extern std::vector<Color> CreateFirePalette(int numColors);

extern std::vector<Color> CreateWaterPalette(int numColors);

extern std::vector<Color> CreateCyclicHSVPpalette(int numColors, int cycles = 1);

extern std::vector<Color> CreateFractalPatternPalette(int numColors);

extern std::vector<Color> CreatePerlinNoisePalette(int numColors, int seed = 0);

extern std::vector<Color> CreateSunsetPalette(int numColors);

extern std::vector<Color> CreatePsychedelicWavePalette(int numColors, float freqH = 3.0f, float freqS = 1.5f, float freqV = 2.0f);

extern std::vector<Color> CreateIceCavePalette(int numColors);

extern std::vector<Color> CreateAccretionDiskPalette(int numColors, float power = 1.f);

extern std::vector<Color> CreateElectricNebulaPalette(int numColors);

extern std::vector<Color> CreateDeepSpaceWideVeinsPalette(int numColors);

extern std::vector<Color> CreateRandomPalette(int numColors);
