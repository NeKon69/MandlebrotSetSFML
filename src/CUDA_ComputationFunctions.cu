#include "CUDA_ComputationFunctions.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

/// Converts HSV (Hue, Saturation, Value) color values to RGB (Red, Green, Blue).
/// This function takes HSV values and outputs the corresponding RGB values
/// as unsigned integers (0-255).
///
/// @param h Hue value (angle in degrees from 0 to 360).
/// @param s Saturation value (range 0 to 1).
/// @param v Value (brightness) (range 0 to 1).
/// @param r Reference to the red channel (output).
/// @param g Reference to the green channel (output).
/// @param b Reference to the blue channel (output).
void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b) {
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

/// Creates a color palette using the HSV color space.
/// The palette transitions smoothly through hues, often used for
/// visualizing the escape count of fractal points.
///
/// @param numColors The number of colors in the palette.
/// @param startHue The starting hue value in degrees.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> createHSVPalette(int numColors, double startHue) {
    std::vector<Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = fmod(startHue + std::pow((t / 0.8), 0.5) * 360.0, 360.0);
        double saturation = 0.5;
        double value = 1.0;

        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a simple black and white color palette.
/// Points inside the set are black, points outside transition to white.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateBlackOWhitePalette(int numColors) {
    std::vector<Color> palette;
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
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a color palette with transitions between specific key colors,
/// including blues, white, orange, and black.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> BluePlusBlackWhitePalette(int numColors) {
    std::vector<Color> palette;

    /// Define control points for the color gradient.
    const double positions[] = { 0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0 };
    const Color colors[] = {
            Color(0, 7, 100),
            Color(32, 107, 203),
            Color(237, 255, 255),
            Color(255, 170, 0),
            Color(0, 2, 0),
            Color(0, 7, 100)
    };
    const int numControlPoints = sizeof(positions) / sizeof(positions[0]);

    palette.reserve(numColors);

    /// Interpolate colors between the control points.
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

        palette.push_back(Color(r, g, b, 255));
    }

    return palette;
}

/// Creates a grayscale palette where the brightness oscillates.
/// This can create interesting banding effects in the fractal visualization.
///
/// @param numColors The number of colors in the palette.
/// @param frequency The frequency of the oscillation.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateOscillatingGrayscalePalette(int numColors, int frequency) {
    std::vector<Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double value = 0.5 * (1 + sin(2 * 3.14159265358979323846 * frequency * t));
        unsigned int c = static_cast<unsigned int>(value * 255);
        palette.push_back(Color(c, c, c, 255));
    }
    return palette;
}

/// Creates a color palette by linearly interpolating between a set of key colors (Blue, Green, Red).
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateInterpolatedPalette(int numColors) {
    std::vector<Color> keyColors = {
            Color(0, 0, 255),
            Color(0, 255, 0),
            Color(255, 0, 0)
    };
    int segments = keyColors.size() - 1;
    std::vector<Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        int segment = static_cast<int>(t * segments);
        if (segment >= segments) segment = segments - 1;
        double localT = (t * segments) - segment;
        Color color1 = keyColors[segment];
        Color color2 = keyColors[segment + 1];
        unsigned int r = static_cast<unsigned int>(color1.r + localT * (color2.r - color1.r));
        unsigned int g = static_cast<unsigned int>(color1.g + localT * (color2.g - color1.g));
        unsigned int b = static_cast<unsigned int>(color1.b + localT * (color2.b - color1.b));
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a color palette with soft, pastel colors.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreatePastelPalette(int numColors) {
    std::vector<Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = t * 360.0;
        double saturation = 0.3 + 0.5 * sin(2 * 3.14159265358979323846 * t);
        double value = 0.7 + 0.3 * cos(2 * 3.14159265358979323846 * t);
        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a color palette that transitions through colors resembling fire (reds, oranges, yellows, white).
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateFirePalette(int numColors) {
    std::vector<Color> palette;
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
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a color palette that transitions through colors resembling water (blues, cyans, white).
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateWaterPalette(int numColors) {
    std::vector<Color> palette;
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
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a cyclic HSV palette, repeating the full hue range multiple times.
/// This creates distinct bands of color in the fractal visualization.
///
/// @param numColors The number of colors in the palette.
/// @param cycles The number of times the hue range repeats.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateCyclicHSVPpalette(int numColors, int cycles) {
    std::vector<Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = fmod(t * 360.0 * cycles, 360.0);
        double saturation = 1.0;
        double value = 1.0;
        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a color palette with a patterned appearance using sine waves
/// to modulate RGB values.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateFractalPatternPalette(int numColors) {
    std::vector<Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        unsigned int r = 255 * (0.5 + 0.5 * sin(20 * t));
        unsigned int g = 255 * (0.5 + 0.5 * sin(30 * t + 2));
        unsigned int b = 255 * (0.5 + 0.5 * sin(40 * t + 4));
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a color palette with variations in hue, saturation, and value
/// modulated by Perlin noise. This can create organic or turbulent color transitions.
///
/// @param numColors The number of colors in the palette.
/// @param seed The seed for the random number generator used in Perlin noise.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreatePerlinNoisePalette(int numColors, int seed) {
    std::vector<Color> palette;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> noise(-0.1, 0.1);

    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;
        double hue = fmod(360 * (t + 0.5 * sin(2 * 3.14159265358979323846 * t * 5)), 360);
        double sat = 0.7 + 0.3 * cos(2 * 3.14159265358979323846 * t * 3 + noise(gen));
        double val = 0.9 + 0.1 * noise(gen);

        unsigned int r, g, b;
        HSVtoRGB(hue, sat, val, r, g, b);
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Creates a multi-gradient color palette inspired by a sunset.
/// Transitions through deep blue, purple, pink, orange, yellow, and white,
/// providing a rich and warm color scheme for fractal visualization.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated sunset palette.
std::vector<Color> CreateSunsetPalette(int numColors) {
    std::vector<Color> palette;

    /// Define control points for the color gradient.
    const double positions[] = { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

    const Color colors[] = {
            Color(0, 0, 128),
            Color(128, 0, 128),
            Color(255, 105, 180),
            Color(255, 165, 0),
            Color(255, 255, 0),
            Color(255, 255, 255)
    };
    const int numSegments = sizeof(positions) / sizeof(positions[0]) - 1;

    /// Interpolate colors between the control points.
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / (numColors - 1);
        int segment = 0;
        while (segment < numSegments && t > positions[segment + 1]) {
            segment++;
        }
        if (segment >= numSegments) segment = numSegments - 1;

        double tSegment = (t - positions[segment]) / (positions[segment + 1] - positions[segment]);
        Color color1 = colors[segment];
        Color color2 = colors[segment + 1];
        unsigned int r = static_cast<unsigned int>(color1.r + tSegment * (color2.r - color1.r));
        unsigned int g = static_cast<unsigned int>(color1.g + tSegment * (color2.g - color1.g));
        unsigned int b = static_cast<unsigned int>(color1.b + tSegment * (color2.b - color1.b));
        palette.push_back(Color(r, g, b, 255));
    }
    return palette;
}

/// Constants for PI and Golden Ratio, used in some palette calculations.
constexpr float PI = 3.14159265f;
constexpr float GOLDEN_RATIO = 1.618033988749895f;

/// Struct to hold HSV color components as floats (0-1 for S,V; 0-360 for H).
struct HSVColor {
    float h;
    float s;
    float v;
};

/// Converts an RGB Color struct to an HSVColor struct.
///
/// @param color The input RGB color.
/// @return HSVColor The converted HSV color.
HSVColor RGBtoHSV(const Color& color) {
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

/// Converts an HSVColor struct to an RGB Color struct.
///
/// @param hsv The input HSV color.
/// @return Color The converted RGB color.
Color HSVtoRGB(const HSVColor& hsv) {
    float h = hsv.h;
    float s = hsv.s;
    float v = hsv.v;
    Color rgb;
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

/// Performs linear interpolation between two float values.
///
/// @param a The start value.
/// @param b The end value.
/// @param t The interpolation factor (0.0 to 1.0).
/// @return float The interpolated value.
float Lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/// Performs linear interpolation between two RGB Color structs.
///
/// @param c1 The start color.
/// @param c2 The end color.
/// @param t The interpolation factor (0.0 to 1.0).
/// @return Color The interpolated color.
Color LerpColorRGB(const Color& c1, const Color& c2, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    Color result;
    result.r = static_cast<unsigned int>(Lerp(static_cast<float>(c1.r), static_cast<float>(c2.r), t));
    result.g = static_cast<unsigned int>(Lerp(static_cast<float>(c1.g), static_cast<float>(c2.g), t));
    result.b = static_cast<unsigned int>(Lerp(static_cast<float>(c1.b), static_cast<float>(c2.b), t));
    result.a = static_cast<unsigned int>(Lerp(static_cast<float>(c1.a), static_cast<float>(c2.a), t));
    return result;
}

/// Performs linear interpolation between two HSVColor structs, then converts back to RGB.
/// Interpolating in HSV space can sometimes produce more perceptually smooth transitions.
///
/// @param c1 The start color (RGB).
/// @param c2 The end color (RGB).
/// @param t The interpolation factor (0.0 to 1.0).
/// @return Color The interpolated color (RGB).
Color LerpColorHSV(const Color& c1, const Color& c2, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    HSVColor hsv1 = RGBtoHSV(c1);
    HSVColor hsv2 = RGBtoHSV(c2);

    float h;
    float d = hsv2.h - hsv1.h;

    /// Handle hue wrapping around 360 degrees for shortest path interpolation.
    if (d > 180.0f)       h = Lerp(hsv1.h + 360.0f, hsv2.h, t);
    else if (d < -180.0f) h = Lerp(hsv1.h, hsv2.h + 360.0f, t);
    else                  h = Lerp(hsv1.h, hsv2.h, t);

    h = fmod(h, 360.0f);

    HSVColor resultHSV;
    resultHSV.h = h;
    resultHSV.s = Lerp(hsv1.s, hsv2.s, t);
    resultHSV.v = Lerp(hsv1.v, hsv2.v, t);

    unsigned int alpha = static_cast<unsigned int>(Lerp(static_cast<float>(c1.a), static_cast<float>(c2.a), t));
    Color resultRGB = HSVtoRGB(resultHSV);
    resultRGB.a = alpha;

    return resultRGB;
}

/// Implements the smoothstep function, providing smooth interpolation
/// between 0 and 1 within a specified range.
///
/// @param edge0 The lower edge of the range.
/// @param edge1 The upper edge of the range.
/// @param x The input value.
/// @return float The smoothly interpolated value.
float Smoothstep(float edge0, float edge1, float x) {
    x = std::max(0.0f, std::min(1.0f, (x - edge0) / (edge1 - edge0)));
    return x * x * (3.0f - 2.0f * x);
}

/// Struct representing a control point in a color gradient,
/// defined by a position (0.0 to 1.0) and a color.
struct ColorNode {
    float position;
    Color color;
};

/// Gets a color from a list of color nodes based on an interpolation factor 't'.
/// It finds the two nodes that 't' falls between and interpolates their colors
/// using the provided interpolator function (e.g., LerpColorRGB or LerpColorHSV).
///
/// @param t The interpolation factor (0.0 to 1.0).
/// @param nodes A sorted vector of ColorNode control points.
/// @param interpolator The function used for color interpolation.
/// @return Color The interpolated color.
Color GetColorFromNodes(float t, const std::vector<ColorNode>& nodes,
                        std::function<Color(const Color&, const Color&, float)> interpolator)
{
    t = std::max(0.0f, std::min(1.0f, t));

    if (nodes.empty()) {
        return Color{0, 0, 0, 255};
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

/// Creates a psychedelic-themed color palette using sine waves to modulate
/// HSV components.
///
/// @param numColors The number of colors in the palette.
/// @param freqH Frequency for hue modulation.
/// @param freqS Frequency for saturation modulation.
/// @param freqV Frequency for value modulation.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreatePsychedelicWavePalette(int numColors, float freqH, float freqS, float freqV) {
    if (numColors <= 0) return {};
    std::vector<Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = Color(253,61,181);
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

/// Creates a color palette inspired by an ice cave, featuring blues, cyans, and whites.
/// Uses smoothstep interpolation and adds subtle brightness modulation.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateIceCavePalette(int numColors) {
    if (numColors <= 0) return {};
    std::vector<Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = Color(200, 240, 255);
        return palette;
    }

    /// Define control points for the ice cave gradient.
    std::vector<ColorNode> nodes = {
            {0.0f,  Color(255, 255, 255)},
            {0.2f,  Color(210, 245, 255)},
            {0.4f,  Color(150, 220, 255)},
            {0.6f,  Color(100, 180, 240)},
            {0.75f, Color(120, 170, 250)},
            {0.9f,  Color(80, 150, 220)},
            {1.0f,  Color(50, 100, 180)}
    };

    std::sort(nodes.begin(), nodes.end(), [](const ColorNode& a, const ColorNode& b) {
        return a.position < b.position;
    });

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float smoothT = Smoothstep(0.0f, 1.0f, t);
        palette[i] = GetColorFromNodes(smoothT, nodes, LerpColorHSV);

        /// Add subtle brightness modulation.
        HSVColor currentHSV = RGBtoHSV(palette[i]);
        float brightnessModulation = 0.05f * std::sin(t * 2.0f * PI * 10.0f);
        currentHSV.v = std::max(0.0f, std::min(1.0f, currentHSV.v + brightnessModulation));
        palette[i] = HSVtoRGB(currentHSV);
    }

    return palette;
}

/// Creates a color palette inspired by an accretion disk, transitioning through
/// reds, oranges, yellows, and blues/whites. Uses smoothstep and power function
/// for non-linear transitions.
///
/// @param numColors The number of colors in the palette.
/// @param power The power factor for non-linear interpolation.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateAccretionDiskPalette(int numColors, float power) {
    if (numColors <= 0) return {};
    std::vector<Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = Color(180, 0, 0);
        return palette;
    }

    /// Define control points for the accretion disk gradient.
    std::vector<ColorNode> nodes = {
            {0.0f,  Color(0, 0, 0)},
            {0.3f,  Color(180, 0, 0)},
            {0.6f,  Color(200, 40, 0)},
            {0.8f,  Color(255, 120, 30)},
            {0.9f,  Color(255, 200, 100)},
            {0.97f, Color(255, 255, 220)},
            {1.0f,  Color(220, 240, 255)}
    };

    std::sort(nodes.begin(), nodes.end(), [](const ColorNode& a, const ColorNode& b) {
        return a.position < b.position;
    });

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);

        float smoothT = Smoothstep(0.0f, 1.0f, t);

        /// Apply a power function for non-linear distribution of colors.
        float nonLinearT = 1.0f - std::pow(1.0f - smoothT, power);

        palette[i] = GetColorFromNodes(nonLinearT, nodes, LerpColorHSV);

        /// Add subtle brightness modulation.
        HSVColor currentHSV = RGBtoHSV(palette[i]);
        float brightnessModulation = 0.03f * std::sin(t * 2.0f * PI * 8.0f);
        currentHSV.v = std::max(0.0f, std::min(1.0f, currentHSV.v + brightnessModulation));
        palette[i] = HSVtoRGB(currentHSV);
    }

    /// Ensure the last color matches the last node's color exactly.
    if (numColors > 1) {
        palette[numColors - 1] = nodes.back().color;
    }

    return palette;
}

/// Creates a color palette inspired by an electric nebula, featuring vibrant
/// blues, purples, pinks, and greens. Uses smoothstep and power function
/// for non-linear transitions and adds modulation to the later colors.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateElectricNebulaPalette(int numColors) {
    if (numColors <= 0) return {};
    std::vector<Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = Color(20, 0, 40);
        return palette;
    }

    /// Define control points for the electric nebula gradient.
    std::vector<ColorNode> nodes = {
            {0.0f,  Color(30, 15, 50)},
            {0.25f, Color(80, 80, 150)},
            {0.5f,  Color(50, 150, 220)},
            {0.65f, Color(150, 50, 200)},
            {0.75f, Color(255, 0, 255)},
            {0.85f, Color(200, 255, 50)},
            {0.95f, Color(255, 200, 150)},
            {1.0f,  Color(255, 255, 200)}
    };

    std::sort(nodes.begin(), nodes.end(), [](const ColorNode& a, const ColorNode& b) {
        return a.position < b.position;
    });

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float smoothT = Smoothstep(0.0f, 1.0f, t);

        /// Apply a power function for non-linear distribution.
        float nonLinearT = 1.0f - std::pow(1.0f - smoothT, 2);

        palette[i] = GetColorFromNodes(nonLinearT, nodes, LerpColorHSV);
    }

    /// Add modulation to the later part of the palette.
    int startModulation = numColors * 6 / 10;
    for (int i = startModulation; i < numColors; ++i) {
        float mod_t = static_cast<float>(i - startModulation) / (numColors - startModulation);
        HSVColor hsv = RGBtoHSV(palette[i]);

        hsv.s = std::min(1.0f, hsv.s + 0.3f * sin(mod_t * 2 * PI));
        hsv.v = std::min(1.0f, hsv.v + 0.2f * sin(mod_t * 3 * PI));

        palette[i] = HSVtoRGB(hsv);
    }

    /// Ensure the last color matches the last node's color exactly.
    if (numColors > 1) {
        palette[numColors - 1] = nodes.back().color;
    }

    return palette;
}

/// Creates a color palette inspired by deep space with wide veins of color.
/// Uses non-linear interpolation and adds occasional white "stars".
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateDeepSpaceWideVeinsPalette(int numColors) {
    if (numColors <= 0) return {};
    std::vector<Color> palette(numColors);
    if (numColors == 1) {
        palette[0] = Color(10, 0, 20);
        return palette;
    }

    /// Define control points for the deep space gradient.
    std::vector<ColorNode> nodes = {
            {0.0f,  Color(20, 0, 40)},
            {0.15f, Color(50, 10, 80)},
            {0.3f,  Color(80, 30, 100)},
            {0.5f,  Color(40, 10, 60)},
            {0.6f,  Color(0, 200, 200)},
            {0.7f,  Color(80, 150, 200)},
            {0.8f,  Color(30, 5, 50)},
            {0.85f, Color(255, 50, 255)},
            {0.92f, Color(200, 100, 200)},
            {1.0f,  Color(20, 0, 40)}
    };

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        /// Apply a non-linear interpolation curve.
        float nonLinearT = t < 0.5 ? 2 * t * t : 1 - pow(-2 * t + 2, 2) / 2;
        palette[i] = GetColorFromNodes(nonLinearT, nodes, LerpColorHSV);

        /// Randomly add white "stars" to the palette.
        if (rand() / (float)RAND_MAX < 0.02f && i > numColors / 4) {
            palette[i] = Color(255, 255, 255, 200);
        }
    }

    return palette;
}

/// Creates a random color palette by generating random parameters for
/// sine wave modulation of HSV components.
///
/// @param numColors The number of colors in the palette.
/// @return std::vector<Color> A vector containing the generated color palette.
std::vector<Color> CreateRandomPalette(int numColors) {
    if (numColors <= 0) return {};

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> freqDist(1.0f, 8.0f);
    std::uniform_real_distribution<float> phaseDist(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> ampDist(0.f, 1.0f); // Amplitude (how much it varies)
    std::uniform_real_distribution<float> baseDist(0.5f, 0.9f); // Base value for S/V
    std::uniform_real_distribution<float> hueBaseDist(0.0f, 360.0f);
    std::uniform_real_distribution<float> hueAmpDist(90.0f, 180.0f); // Hue variation range

    float freqH = freqDist(gen);
    float phaseH = phaseDist(gen);
    float baseH = hueBaseDist(gen);
    float ampH = hueAmpDist(gen);

    float freqS = freqDist(gen);
    float phaseS = phaseDist(gen);
    float baseS = baseDist(gen);
    float ampS = ampDist(gen);

    float freqV = freqDist(gen);
    float phaseV = phaseDist(gen);
    float baseV = baseDist(gen);
    float ampV = ampDist(gen);

    std::vector<Color> palette(numColors);
    if (numColors == 1) {
        float h = fmod(baseH + ampH * std::sin(phaseH) + 360.0f, 360.0f);
        float s = std::max(0.0f, std::min(1.0f, baseS + ampS * std::sin(phaseS)));
        float v = std::max(0.0f, std::min(1.0f, baseV + ampV * std::sin(phaseV)));
        palette[0] = HSVtoRGB({ h, s, v });
        return palette;
    }

    for (int i = 0; i < numColors; ++i) {
        float t = static_cast<float>(i) / (numColors - 1);
        float angle = t * 2.0f * PI;

        float h = baseH + ampH * std::sin(angle * freqH + phaseH);
        h = fmod(h + 360.0f, 360.0f);

        float s = baseS + ampS * std::sin(angle * freqS + phaseS);
        s = std::max(0.0f, std::min(1.0f, s));

        float v = baseV + ampV * std::sin(angle * freqV + phaseV);
        v = std::max(0.0f, std::min(1.0f, v));

        palette[i] = HSVtoRGB({ h, s, v });
    }

    return palette;
}

/// CUDA kernel for applying 4x Super-Sampling Anti-Aliasing (SSAA).
/// It takes a high-resolution source image (src) and downsamples it
/// to a lower-resolution destination image (dest) by averaging 2x2 blocks
/// of pixels from the source.
///
/// @param src Pointer to the high-resolution source image data (device memory).
/// @param dest Pointer to the low-resolution destination image data (device memory).
/// @param src_width Width of the source image.
/// @param src_height Height of the source image.
/// @param dest_width Width of the destination image.
/// @param dest_height Height of the destination image.
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