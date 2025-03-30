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
inline static void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b) {
    h = fmod(h, 360.0);
    double c = v * s;
    double x = c * (1 - std::fabs(fmod(h / 60.0, 2) - 1));
    double m = v - c;

    double r_, g_, b_;
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
std::vector<sf::Color> createHSVPalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;

        double hue;
        double saturation = 1.0;
        double value;

        hue = std::pow((t - 0.2) / 0.8, 0.5) * 360.0;
        value = 1.0;

        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

std::vector<sf::Color> CreateBlackOWhitePalette(int numColors) {
	std::vector<sf::Color> palette;
	for (int i = 0; i < numColors; ++i) {
		double t = static_cast<double>(i) / numColors;
		unsigned int r, g, b;
		r = static_cast<unsigned int>(t * 255);
		g = static_cast<unsigned int>(t * 255);
		b = static_cast<unsigned int>(t * 255);
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

std::vector<sf::Color> palette(CreateBlackOWhitePalette(200000));

