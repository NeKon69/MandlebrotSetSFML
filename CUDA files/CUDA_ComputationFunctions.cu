#include "CUDA_ComputationFunctions.cuh"
#include <cuda_runtime.h>  
#include <vector>
#include <cmath>
#include <iostream>


inline static void cudaCheckError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}


/**
 * @brief Converts HSV (Hue, Saturation, Value) color values to RGB (Red, Green, Blue).
 * This function performs the conversion based on the provided HSV values and
 * calculates the corresponding RGB values, which are then stored in the provided
 * unsigned integer references for red, green, and blue channels.
 *
 * @param h Hue value (angle in degrees from 0 to 360).
 * @param s Saturation value (range 0 to 1).
 * @param v Value (brightness) (range 0 to 1).
 * @param r Reference to the red channel (output).
 * @param g Reference to the green channel (output).
 * @param b Reference to the blue channel (output).
 */
[[assume(h > 0 && h < 360 && s >= 0 && s <= 1 && v >= 0 && v <= 1 && r >= 0 && g >= 0 && b >= 0 && r <= 255 && g <= 255 && b <= 255)]] 
inline static void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b) {
    h = fmod(h, 360.0);
    double c = v * s;
    double x = c * (1 - std::fabs(fmod(h / 60.0, 2) - 1));
    double m = v - c;

    double r_ = 0, g_ = 0, b_ = 0;
    if (h < 60) { r_ = c, g_ = x, b_ = 0; }
    else if (h < 120) { r_ = x, g_ = c, b_ = 0; }
    else if (h < 180) { r_ = 0, g_ = c, b_ = x; }
    else if (h < 240) { r_ = 0, g_ = x, b_ = c; }
    else if (h < 300) { r_ = x, g_ = 0, b_ = c; }
    else { r_ = c, g_ = 0, b_ = x; }

    r = static_cast<unsigned int>((r_ + m) * 255);
    g = static_cast<unsigned int>((g_ + m) * 255);
    b = static_cast<unsigned int>((b_ + m) * 255);
}

/**
 * @brief Creates a color palette using HSV color space and converts it to RGB.
 * The palette is designed to smoothly transition colors for visualization of the Mandelbrot set.
 * @param numColors The number of colors in the palette.
 * @return std::vector<sf::Color> A vector containing the generated color palette.
 **/
std::vector<sf::Color> createHSVPalette(int numColors, double startHue) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = fmod(startHue + std::pow((t / 0.8), 0.5) * 360.0, 360.0);
        double saturation = 0.5;
        double value = 1.0;

        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

std::vector<sf::Color> CreateBlackOWhitePalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        unsigned int r, g, b;
        if (i == numColors - 1) {
            r = g = b = 0;
        }
        else if (i >= numColors - 3) {
            r = g = b = 255;
        }
        else {
            r = g = b = static_cast<unsigned int>(128 + 127 * t);
        }
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

std::vector<sf::Color> BluePlusBlackWhitePalette(int numColors) {
    std::vector<sf::Color> palette;

    const double positions[] = { 0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0 };
    const sf::Color colors[] = {
        sf::Color(0, 7, 100),
        sf::Color(32, 107, 203),
        sf::Color(237, 255, 255),
        sf::Color(255, 170, 0),
        sf::Color(0, 2, 0), 
        sf::Color(0, 7, 100)
    };
    const int numControlPoints = sizeof(positions) / sizeof(positions[0]);

    palette.reserve(numColors);

    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);

        int segment = 0;
        for (int j = 0; j < numControlPoints - 1; ++j) {
            if (t >= positions[j] && t <= positions[j + 1]) {
                segment = j;
                break;
            }
        }

        double tSegment = (t - positions[segment]) / (positions[segment + 1] - positions[segment]);
        unsigned int r = static_cast<unsigned int>(
            colors[segment].r * (1.0 - tSegment) + colors[segment + 1].r * tSegment
            );
        unsigned int g = static_cast<unsigned int>(
            colors[segment].g * (1.0 - tSegment) + colors[segment + 1].g * tSegment
            );
        unsigned int b = static_cast<unsigned int>(
            colors[segment].b * (1.0 - tSegment) + colors[segment + 1].b * tSegment
            );

        r = std::min(std::max(r, 0u), 255u);
        g = std::min(std::max(g, 0u), 255u);
        b = std::min(std::max(b, 0u), 255u);

        palette.push_back(sf::Color(r, g, b, 255));
    }

    return palette;
}

std::vector<sf::Color> CreateOscillatingGrayscalePalette(int numColors, int frequency) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double value = 0.5 * (1 + sin(2 * 3.14159265358979323846 * frequency * t));
        unsigned int c = static_cast<unsigned int>(value * 255);
        palette.push_back(sf::Color(c, c, c, 255));
    }
    return palette;
}

std::vector<sf::Color> CreateInterpolatedPalette(int numColors) {
    std::vector<sf::Color> keyColors = {
        sf::Color(0, 0, 255),
        sf::Color(0, 255, 0),
        sf::Color(255, 0, 0)
    };
    int segments = keyColors.size() - 1;
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        int segment = static_cast<int>(t * segments);
        if (segment >= segments) segment = segments - 1;
        double localT = (t * segments) - segment;
        sf::Color color1 = keyColors[segment];
        sf::Color color2 = keyColors[segment + 1];
        unsigned int r = static_cast<unsigned int>(color1.r + localT * (color2.r - color1.r));
        unsigned int g = static_cast<unsigned int>(color1.g + localT * (color2.g - color1.g));
        unsigned int b = static_cast<unsigned int>(color1.b + localT * (color2.b - color1.b));
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

std::vector<sf::Color> CreatePastelPalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = t * 360.0;
        double saturation = 0.3 + 0.5 * sin(2 * 3.14159265358979323846 * t);
        double value = 0.7 + 0.3 * cos(2 * 3.14159265358979323846 * t);
        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}


std::vector<sf::Color> CreateFirePalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        unsigned int r, g, b;
        if (t < 0.25) {
            r = static_cast<unsigned int>(255 * (t / 0.25));
            g = b = 0;
        }
        else if (t < 0.5) {
            r = 255;
            g = static_cast<unsigned int>(255 * ((t - 0.25) / 0.25));
            b = 0;
        }
        else if (t < 0.75) {
            r = g = 255;
            b = static_cast<unsigned int>(255 * ((t - 0.5) / 0.25));
        }
        else {
            r = g = b = 255;
        }
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}


std::vector<sf::Color> CreateCyclicHSVPpalette(int numColors, int cycles) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = fmod(t * 360.0 * cycles, 360.0);
        double saturation = 1.0;
        double value = 1.0;
        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

/**
 * @brief Applies 4x Super-Sampling Anti-Aliasing (SSAA) to reduce aliasing artifacts.
 * This function takes a high-resolution image (2x width and height of the desired output)
 * and downsamples it to produce a smoother, anti-aliased image. It averages the color
 * of 2x2 pixel blocks in the high-resolution image to generate each pixel in the
 * low-resolution image.
 *
 * @param high_qual The high-resolution input image.
 * @param size The scaling factor (should be 2 for 4x SSAA).
 * @return sf::Image The anti-aliased, low-resolution image.
 */
__global__ 
void ANTIALIASING_SSAA4(unsigned char* src, unsigned char* dest, int src_width, int src_height, int dest_width, int dest_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dest_width && y < dest_height) {
        int r = 0, g = 0, b = 0;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int src_x = x * 2 + i;
                int src_y = y * 2 + j;
                if (src_x < src_width && src_y < src_height) {
                    int src_index = (src_y * src_width + src_x) * 4;
                    r += src[src_index];
                    g += src[src_index + 1];
                    b += src[src_index + 2];
                }
            }
        }
        int dest_index = (y * dest_width + x) * 4;
        dest[dest_index] = r / 4;
        dest[dest_index + 1] = g / 4;
        dest[dest_index + 2] = b / 4;
        dest[dest_index + 3] = 255;
    }
}

