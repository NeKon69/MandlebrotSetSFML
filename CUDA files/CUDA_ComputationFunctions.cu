#include "CUDA_ComputationFunctions.cuh"
#include <cuda_runtime.h>  
#include <vector>
#include <cmath>
#include <iostream>
#include <random>


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

std::vector<sf::Color> CreateWaterPalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        unsigned int r, g, b;
        if (t < 0.25) {
            g = 0;
            r = static_cast<unsigned int>(255 * (t / 0.5));
            b = static_cast<unsigned int>(255 * (t / 0.25));
        }
        else if (t < 0.5) {
            r = 172;
            g = static_cast<unsigned int>(255 * ((t - 0.25) / 0.25));
            b = 255;
        }
        else if (t < 0.75) {
            g = b = 255;
            r = static_cast<unsigned int>(255 * ((t - 0.5) / 0.125));
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

std::vector<sf::Color> CreateFractalPatternPalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        unsigned int r = 255 * (0.5 + 0.5 * sin(20 * t));
        unsigned int g = 255 * (0.5 + 0.5 * sin(30 * t + 2));
        unsigned int b = 255 * (0.5 + 0.5 * sin(40 * t + 4));
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

std::vector<sf::Color> CreatePerlinNoisePalette(int numColors, int seed) {
    std::vector<sf::Color> palette;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> noise(-0.1, 0.1);

    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = fmod(360 * (t + 0.5 * sin(2 * 3.14159265358979323846 * t * 5)), 360);
        double sat = 0.7 + 0.3 * cos(2 * 3.14159265358979323846 * t * 3 + noise(gen));
        double val = 0.9 + 0.1 * noise(gen);

        unsigned int r, g, b;
        HSVtoRGB(hue, sat, val, r, g, b);
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

/**
 * @brief Creates a multi-gradient color palette inspired by a sunset.
 * Transitions through deep blue, purple, pink, orange, yellow, and white,
 * providing a rich and warm color scheme for fractal visualization.
 * @param numColors The number of colors in the palette.
 * @return std::vector<sf::Color> A vector containing the generated sunset palette.
 */
std::vector<sf::Color> CreateSunsetPalette(int numColors) {
    std::vector<sf::Color> palette;

    const double positions[] = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

    const sf::Color colors[] = {
        sf::Color(0, 0, 128),
        sf::Color(128, 0, 128),
        sf::Color(255, 105, 180),
        sf::Color(255, 165, 0),
        sf::Color(255, 255, 0),
        sf::Color(255, 255, 255)
    };
    const int numSegments = sizeof(positions) / sizeof(positions[0]) - 1;

    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        int segment = 0;
        while (segment < numSegments && t > positions[segment + 1]) {
            segment++;
        }
        if (segment >= numSegments) segment = numSegments - 1;

        double tSegment = (t - positions[segment]) / (positions[segment + 1] - positions[segment]);
        sf::Color color1 = colors[segment];
        sf::Color color2 = colors[segment + 1];
        unsigned int r = static_cast<unsigned int>(color1.r + tSegment * (color2.r - color1.r));
        unsigned int g = static_cast<unsigned int>(color1.g + tSegment * (color2.g - color1.g));
        unsigned int b = static_cast<unsigned int>(color1.b + tSegment * (color2.b - color1.b));
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}


constexpr float PI = 3.14159265f;
constexpr float GOLDEN_RATIO = 1.618033988749895f;

struct HSVColor {
    float h;
    float s;
    float v;
};

HSVColor RGBtoHSV(const sf::Color& color) {
    float r = color.r / 255.0f;
    float g = color.g / 255.0f;
    float b = color.b / 255.0f;

    float maxC = std::max({ r, g, b });
    float minC = std::min({ r, g, b });
    float delta = maxC - minC;

    HSVColor hsv;
    hsv.v = maxC;

    if (delta < 0.00001f) {
        hsv.s = 0;
        hsv.h = 0;
        return hsv;
    }
    if (maxC > 0.0f) {
        hsv.s = (delta / maxC);
    }
    else {
        hsv.s = 0;
        hsv.h = 0;
        return hsv;
    }

    if (r >= maxC)
        hsv.h = (g - b) / delta;
    else if (g >= maxC)
        hsv.h = 2.0f + (b - r) / delta;
    else
        hsv.h = 4.0f + (r - g) / delta;

    hsv.h *= 60.0f;

    if (hsv.h < 0.0f)
        hsv.h += 360.0f;

    return hsv;
}

sf::Color HSVtoRGB(const HSVColor& hsv) {
    float h = hsv.h;
    float s = hsv.s;
    float v = hsv.v;
    sf::Color rgb;
    rgb.a = 255;

    if (s <= 0.0f) {
        rgb.r = static_cast<unsigned int>(v * 255);
        rgb.g = static_cast<unsigned int>(v * 255);
        rgb.b = static_cast<unsigned int>(v * 255);
        return rgb;
    }

    float hh = h;
    if (hh >= 360.0f) hh = 0.0f;
    hh /= 60.0f;
    long i = (long)hh;
    float ff = hh - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - (s * ff));
    float t = v * (1.0f - (s * (1.0f - ff)));

    float r, g, b;

    switch (i) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5:
    default: r = v; g = p; b = q; break;
    }

    rgb.r = static_cast<unsigned int>(r * 255);
    rgb.g = static_cast<unsigned int>(g * 255);
    rgb.b = static_cast<unsigned int>(b * 255);
    return rgb;
}

float Lerp(float a, float b, float t) {
    return a + t * (b - a);
}

sf::Color LerpColorRGB(const sf::Color& c1, const sf::Color& c2, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    sf::Color result;
    result.r = static_cast<unsigned int>(Lerp(static_cast<float>(c1.r), static_cast<float>(c2.r), t));
    result.g = static_cast<unsigned int>(Lerp(static_cast<float>(c1.g), static_cast<float>(c2.g), t));
    result.b = static_cast<unsigned int>(Lerp(static_cast<float>(c1.b), static_cast<float>(c2.b), t));
    result.a = static_cast<unsigned int>(Lerp(static_cast<float>(c1.a), static_cast<float>(c2.a), t));
    return result;
}

sf::Color LerpColorHSV(const sf::Color& c1, const sf::Color& c2, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    HSVColor hsv1 = RGBtoHSV(c1);
    HSVColor hsv2 = RGBtoHSV(c2);

    float h;
    float d = hsv2.h - hsv1.h;

    if (d > 180.0f)       h = Lerp(hsv1.h + 360.0f, hsv2.h, t);
    else if (d < -180.0f) h = Lerp(hsv1.h, hsv2.h + 360.0f, t);
    else                  h = Lerp(hsv1.h, hsv2.h, t);

    h = fmod(h, 360.0f);

    HSVColor resultHSV;
    resultHSV.h = h;
    resultHSV.s = Lerp(hsv1.s, hsv2.s, t);
    resultHSV.v = Lerp(hsv1.v, hsv2.v, t);

    unsigned int alpha = static_cast<unsigned int>(Lerp(static_cast<float>(c1.a), static_cast<float>(c2.a), t));
    sf::Color resultRGB = HSVtoRGB(resultHSV);
    resultRGB.a = alpha;

    return resultRGB;
}

float Smoothstep(float edge0, float edge1, float x) {
    x = std::max(0.0f, std::min(1.0f, (x - edge0) / (edge1 - edge0)));
    return x * x * (3.0f - 2.0f * x);
}

struct ColorNode {
    float position;
    sf::Color color;
};

sf::Color GetColorFromNodes(float t, const std::vector<ColorNode>& nodes,
    std::function<sf::Color(const sf::Color&, const sf::Color&, float)> interpolator)
{
    t = std::max(0.0f, std::min(1.0f, t));

    if (nodes.empty()) {
        return sf::Color::Black;
    }
    if (nodes.size() == 1 || t <= nodes.front().position) {
        return nodes.front().color;
    }
    if (t >= nodes.back().position) {
        return nodes.back().color;
    }

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        if (t >= nodes[i].position && t <= nodes[i + 1].position) {
            float segmentT = (t - nodes[i].position) / (nodes[i + 1].position - nodes[i].position);
            return interpolator(nodes[i].color, nodes[i + 1].color, segmentT);
        }
    }

    return nodes.back().color;
}

std::vector<sf::Color> CreatePsychedelicWavePalette(int numColors, float freqH, float freqS, float freqV) {
    if (numColors <= 0) return {};
    std::vector<sf::Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = sf::Color::Magenta;
        return palette;
    }

    float phaseH = 0.0f;
    float phaseS = PI / 2.0f;
    float phaseV = PI;

    float baseHue = 180.0f;
    float hueRange = 180.0f;

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float angle = t * 2.0f * PI;

        float h = baseHue + hueRange * std::sin(angle * freqH + phaseH);
        h = fmod(h + 360.0f, 360.0f);

        float s = 0.75f + 0.25f * std::cos(angle * freqS + phaseS);
        s = std::max(0.0f, std::min(1.0f, s));

        float v = 0.8f + 0.2f * std::sin(angle * freqV + phaseV);
        v = std::max(0.0f, std::min(1.0f, v));

        palette[i] = HSVtoRGB({ h, s, v });
    }

    return palette;
}

std::vector<sf::Color> CreateIceCavePalette(int numColors) {
    if (numColors <= 0) return {};
    std::vector<sf::Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = sf::Color(200, 240, 255);
        return palette;
    }

    std::vector<ColorNode> nodes = {
        {0.0f,  sf::Color(255, 255, 255)},
        {0.2f,  sf::Color(210, 245, 255)},
        {0.4f,  sf::Color(150, 220, 255)},
        {0.6f,  sf::Color(100, 180, 240)},
        {0.75f, sf::Color(120, 170, 250)},
        {0.9f,  sf::Color(80, 150, 220)},
        {1.0f,  sf::Color(50, 100, 180)}
    };

    std::sort(nodes.begin(), nodes.end(), [](const ColorNode& a, const ColorNode& b) {
        return a.position < b.position;
        });

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float smoothT = Smoothstep(0.0f, 1.0f, t);
        palette[i] = GetColorFromNodes(smoothT, nodes, LerpColorHSV);

        HSVColor currentHSV = RGBtoHSV(palette[i]);
        float brightnessModulation = 0.05f * std::sin(t * 2.0f * PI * 10.0f);
        currentHSV.v = std::max(0.0f, std::min(1.0f, currentHSV.v + brightnessModulation));
        palette[i] = HSVtoRGB(currentHSV);
    }

    return palette;
}

std::vector<sf::Color> CreateAccretionDiskPalette(int numColors, float power) {
    if (numColors <= 0) return {};
    std::vector<sf::Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = sf::Color(180, 0, 0);
        return palette;
    }

    std::vector<ColorNode> nodes = {
        {0.0f,  sf::Color(0, 0, 0)}, 
        {0.3f,  sf::Color(180, 0, 0)},
        {0.6f,  sf::Color(200, 40, 0)},
        {0.8f,  sf::Color(255, 120, 30)},
        {0.9f,  sf::Color(255, 200, 100)},
        {0.97f, sf::Color(255, 255, 220)},
        {1.0f,  sf::Color(220, 240, 255)}
    };

    std::sort(nodes.begin(), nodes.end(), [](const ColorNode& a, const ColorNode& b) {
        return a.position < b.position;
        });

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);

        float smoothT = Smoothstep(0.0f, 1.0f, t);

        float nonLinearT = 1.0f - std::pow(1.0f - smoothT, power);

        palette[i] = GetColorFromNodes(nonLinearT, nodes, LerpColorHSV);

        HSVColor currentHSV = RGBtoHSV(palette[i]);
        float brightnessModulation = 0.03f * std::sin(t * 2.0f * PI * 8.0f);
        currentHSV.v = std::max(0.0f, std::min(1.0f, currentHSV.v + brightnessModulation));
        palette[i] = HSVtoRGB(currentHSV);
    }

    if (numColors > 1) {
        palette[numColors - 1] = nodes.back().color;
    }

    return palette;
}


std::vector<sf::Color> CreateElectricNebulaPalette(int numColors) {
    if (numColors <= 0) return {};
    std::vector<sf::Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = sf::Color(20, 0, 40);
        return palette;
    }

    std::vector<ColorNode> nodes = {
        {0.0f,  sf::Color(30, 15, 50)},
        {0.25f, sf::Color(80, 80, 150)},
        {0.5f,  sf::Color(50, 150, 220)},
        {0.65f, sf::Color(150, 50, 200)},
        {0.75f, sf::Color(255, 0, 255)},
        {0.85f, sf::Color(200, 255, 50)},
        {0.95f, sf::Color(255, 200, 150)},
        {1.0f,  sf::Color(255, 255, 200)}
    };

    std::sort(nodes.begin(), nodes.end(), [](const ColorNode& a, const ColorNode& b) {
        return a.position < b.position;
        });

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float smoothT = Smoothstep(0.0f, 1.0f, t);

        float nonLinearT = 1.0f - std::pow(1.0f - smoothT, 2);

        palette[i] = GetColorFromNodes(nonLinearT, nodes, LerpColorHSV);
    }

    int startModulation = numColors * 6 / 10;
    for (int i = startModulation; i < numColors; ++i) {
        float mod_t = static_cast<float>(i - startModulation) / (numColors - startModulation);
        HSVColor hsv = RGBtoHSV(palette[i]);

        hsv.s = std::min(1.0f, hsv.s + 0.3f * sin(mod_t * 2 * PI));
        hsv.v = std::min(1.0f, hsv.v + 0.2f * sin(mod_t * 3 * PI));

        palette[i] = HSVtoRGB(hsv);
    }

    if (numColors > 1) {
        palette[numColors - 1] = nodes.back().color;
    }

    return palette;
}


std::vector<sf::Color> CreateDeepSpaceWideVeinsPalette(int numColors) {
    if (numColors <= 0) return {};
    std::vector<sf::Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = sf::Color(10, 0, 20);
        return palette;
    }

    std::vector<ColorNode> nodes = {
        {0.0f,  sf::Color(20, 0, 40)},
        {0.15f, sf::Color(50, 10, 80)},
        {0.3f,  sf::Color(80, 30, 100)},
        {0.5f,  sf::Color(40, 10, 60)},
        {0.6f,  sf::Color(0, 200, 200)},
        {0.7f,  sf::Color(80, 150, 200)},
        {0.8f,  sf::Color(30, 5, 50)},
        {0.85f, sf::Color(255, 50, 255)},
        {0.92f, sf::Color(200, 100, 200)},
        {1.0f,  sf::Color(20, 0, 40)}
    };

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float nonLinearT = t < 0.5 ? 2 * t * t : 1 - pow(-2 * t + 2, 2) / 2;
        palette[i] = GetColorFromNodes(nonLinearT, nodes, LerpColorHSV);

        if (rand() / (float)RAND_MAX < 0.02f && i > numColors / 4) {
            palette[i] = sf::Color(255, 255, 255, 200);
        }
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

