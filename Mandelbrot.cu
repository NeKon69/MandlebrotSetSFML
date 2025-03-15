#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <complex>
#include <cmath>
#include <vector_alias.hpp>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <thread>


/**
 * @brief Maximum number of iterations for Mandelbrot set calculation.
 * Higher values increase detail but also computation time.
 */
unsigned int max_iterations = 2000;

double basic_zoom_x = 240.0f;
double basic_zoom_y = 240.0f;

double zoom_x = basic_zoom_x;
double zoom_y = basic_zoom_y;

double x_offset = 3.5f;
double y_offset = 2.5f;

double zoom_factor = 1.0f;
double zoom_speed = 0.1f;

sf::Vector2i drag_start_pos;
bool is_dragging = false;

unsigned int width = 1600;
unsigned int height = 1200;

/**
 * @brief Pixel buffer in host memory to store the rendered image data.
 */
unsigned char* pixels = new unsigned char[width * height * 4];

/**
 * @brief Pixel buffer in device (GPU) memory, used for CUDA kernel.
 */
unsigned char* d_pixels;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ double complex_mult_real(double real1, double imag1, double real2, double imag2) {
    return real1 * real2 - imag1 * imag2;
}

__device__ double complex_mult_imag(double real1, double imag1, double real2, double imag2) {
    return real1 * imag2 + imag1 * real2;
}

__device__ double dev_abs(double x) {
    return (x >= 0) ? x : -x;
}

__device__ double dev_log2(double x) {
    return log(x) / log(2.0);
}

__device__ double dev_sqrt(double x) {
    return sqrt(x);
}

__device__ double complex_abs2(double real, double imag) {
    return real * real + imag * imag;
}

__device__ double Gradient(double current_iteration, double max_iter) {
    if (current_iteration >= max_iter) return 0.0;
    current_iteration = dev_sqrt(current_iteration);
    max_iter = dev_sqrt(max_iter);
    return (current_iteration) / static_cast<double>(max_iter);
}

__device__ sf::Color getPaletteColor(int index, int paletteSize, sf::Color* palette) {
    index = (index < 0) ? 0 : ((index >= paletteSize) ? paletteSize - 1 : index);
    return palette[index];
}


void HSVtoRGB(double h, double s, double v, unsigned int& r, unsigned int& g, unsigned int& b) {
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
 */
std::vector<sf::Color> createHSVPalette(int numColors) {
    std::vector<sf::Color> palette;
    for (int i = 0; i < numColors; ++i) {
        double t = static_cast<double>(i) / numColors;

        double hue;
        double saturation = 1.0;
        double value;

        if (t < 0.2) {
            hue = 240.0;
            value = 0.3 + t * (1.0 - 0.3) / 0.2;
        }
        else {
            hue = std::pow((t - 0.2) / 0.8, 0.5) * 360.0;
            value = 1.0;
        }

        unsigned int r, g, b;
        HSVtoRGB(hue, saturation, value, r, g, b);
        palette.push_back(sf::Color(r, g, b, 255));
    }
    return palette;
}

std::vector<sf::Color> palette = createHSVPalette(200000);
sf::Image image({ 1600, 1200 }, sf::Color::Black);


/**
 * @brief CUDA kernel function to calculate and render the Mandelbrot set.
 * This kernel is executed in parallel by multiple threads on the GPU.
 * Each thread calculates the color of a single pixel based on its position
 * and the Mandelbrot set algorithm.
 *
 * @param pixels Pointer to the pixel data buffer in device memory.
 * @param width Image width.
 * @param height Image height.
 * @param zoom_x Zoom level along the x-axis.
 * @param zoom_y Zoom level along the y-axis.
 * @param x_offset Offset in the x-direction to move the view.
 * @param y_offset Offset in the y-direction to move the view.
 * @param d_palette Color palette in device memory to color the Mandelbrot set.
 * @param paletteSize Size of the color palette.
 * @param maxIterations Maximum iterations for Mandelbrot calculation.
 */
__global__ void mandelbrot(unsigned char* pixels, int width, int height, double zoom_x, double zoom_y, double x_offset, double y_offset, sf::Color* d_palette, int paletteSize, double maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double real = x / zoom_x - x_offset;
        double imag = y / zoom_y - y_offset;
        double z_real = 0.0;
        double z_imag = 0.0;
        double current_iteration = 0;

        while (complex_abs2(z_real, z_imag) < 4 && current_iteration < maxIterations) {
            double new_real = complex_mult_real(z_real, z_imag, z_real, z_imag) + real;
            double new_imag = complex_mult_imag(z_real, z_imag, z_real, z_imag) + imag;

            // Some strange thing
            //double new_real = complex_mult_real(z_real, z_imag, z_real, z_imag) * new_real + real;
            //double new_imag = complex_mult_imag(z_real, z_imag, z_real, z_imag) * new_imag + imag;
            z_real = new_real;
            z_imag = new_imag;
            current_iteration++;
        }

        unsigned char r, g, b;
        if (current_iteration == maxIterations) {
            r = g = b = 0;
        }
        else {
            // Color gradient calculation based on iteration count for points outside the set
            double max_iters = maxIterations;

            // Smooth iteration count
            current_iteration = current_iteration + 1 - dev_log2(dev_log2(dev_abs(dev_sqrt(complex_abs2(z_real, z_imag)))));

            // Calculate gradient value
            double gradient = Gradient(current_iteration, max_iters);

            // Map gradient to palette index
            int index = static_cast<int>(gradient * (paletteSize - 1));
            sf::Color color = getPaletteColor(index, paletteSize, d_palette);
            r = color.r;
            g = color.g;
            b = color.b;
        }
        int index = (y * width + x) * 4;
        pixels[index + 0] = r;
        pixels[index + 1] = g;
        pixels[index + 2] = b;
        pixels[index + 3] = 255;
    }
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
static sf::Image ANTIALIASING_SSAA4(sf::Image& high_qual, unsigned int size) {
    sf::Image low_qual({ high_qual.getSize().x / size * 2, high_qual.getSize().y / size * 2 }, sf::Color::Black);
    for (unsigned int x = 0; x < high_qual.getSize().x; x += 2) {
        for (unsigned int y = 0; y < high_qual.getSize().y; y += 2) {
            sf::Color colorTopLeft = high_qual.getPixel({ x, y });
            sf::Color colorTopRight = high_qual.getPixel({ x + 1, y });
            sf::Color colorBottomLeft = high_qual.getPixel({ x, y + 1 });
            sf::Color colorBottomRight = high_qual.getPixel({ x + 1, y + 1 });
            sf::Color color = sf::Color((colorTopLeft.r + colorTopRight.r + colorBottomLeft.r + colorBottomRight.r) / 4, (colorTopLeft.g + colorTopRight.g + colorBottomLeft.g + colorBottomRight.g) / 4, (colorTopLeft.b + colorTopRight.b + colorBottomLeft.b + colorBottomRight.b) / 4);
            low_qual.setPixel({ x / 2, y / 2 }, color);
        }
    }
    return low_qual;
}


/**
 * @brief Handles zoom functionality based on mouse wheel input.
 * Adjusts the zoom factor and offsets to zoom in or out centered around the mouse position.
 *
 * @param wheel_delta The direction and magnitude of the mouse wheel scroll.
 * @param mouse_pos The current mouse position in window coordinates.
 */
void handleZoom(float wheel_delta, const sf::Vector2i mouse_pos) {
    double old_zoom_x = zoom_x;
    double old_zoom_y = zoom_y;
    double old_x_offset = x_offset;
    double old_y_offset = y_offset;

    double zoom_change = 1.0 + wheel_delta * zoom_speed;
    zoom_factor *= zoom_change; // Update zoom factor based on wheel delta

    zoom_factor = std::max(std::min(zoom_factor, 1000000000000.0), 0.01);

    zoom_x = basic_zoom_x * zoom_factor;
    zoom_y = basic_zoom_y * zoom_factor;

    // Calculate the change in offset to keep the zoom centered at the mouse position
    double image_mouse_x = mouse_pos.x * (1600.0 / 800.0);
    double image_mouse_y = mouse_pos.y * (1200.0 / 600.0);

    x_offset = old_x_offset + (image_mouse_x / zoom_x - image_mouse_x / old_zoom_x);
    y_offset = old_y_offset + (image_mouse_y / zoom_y - image_mouse_y / old_zoom_y);

    std::cout << "Zoom Factor: " << zoom_factor << ", Offset: (" << x_offset << ", " << y_offset << ")" << std::endl;
}


/**
 * @brief Starts the dragging operation when the mouse button is pressed.
 * Records the starting mouse position.
 *
 * @param mouse_pos The current mouse position.
 */
void start_dragging(sf::Vector2i mouse_pos) {
    is_dragging = true;
    drag_start_pos = mouse_pos;
}


/**
 * @brief Handles the dragging operation while the mouse is moved and the button is held down.
 * Updates the view offset based on the mouse movement since the drag started.
 *
 * @param mouse_pos The current mouse position.
 */
void dragging(sf::Vector2i mouse_pos) {
    if (!is_dragging) return;

    sf::Vector2i delta_pos = mouse_pos - drag_start_pos;

    double delta_real = static_cast<double>(delta_pos.x) / zoom_x;
    double delta_imag = static_cast<double>(delta_pos.y) / zoom_y;

    x_offset += delta_real;
    y_offset += delta_imag;

    drag_start_pos = mouse_pos;
}



/**
 * @brief Stops the dragging operation when the mouse button is released.
 * Resets the dragging flag.
 */
void stop_dragging() {
    is_dragging = false;
}



int main() {
	// Used to render or not the image based on the moves made with the mouse
    bool need_render = true;
    sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Mandelbrot");
    sf::Image image({ 1600, 1200 }, sf::Color::Black);
    sf::Image compressed({ 800, 600 }, sf::Color::Black);
    sf::Clock timer;

    double max_iters = max_iterations;

    sf::Vector2i mouse;

    // Palette transfer to device memory for CUDA kernel access
    sf::Color* d_palette;
    int paletteSize = palette.size();
    cudaMalloc(&d_palette, paletteSize * sizeof(sf::Color)); // Allocate device memory for palette
    cudaMemcpy(d_palette, palette.data(), paletteSize * sizeof(sf::Color), cudaMemcpyHostToDevice); // Copy palette data to device

    cudaMalloc(&d_pixels, width * height * 4 * sizeof(unsigned char)); // Allocate device memory for pixels

    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (const auto* mm = event->getIf<sf::Event::MouseMoved>()) mouse = mm->position;

            if (event->is<sf::Event::Closed>()) {
                window.close();
            }

            if (const auto* button = event->getIf<sf::Event::KeyPressed>()) {
                if (button->scancode == sf::Keyboard::Scancode::Escape) {
                    window.close();
                }
            }

            if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
                need_render = true;
                handleZoom(mouseWheelScrolled->delta, mouse);
            }

            if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    start_dragging(mouse);
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleased->button == sf::Mouse::Button::Left) {
                    stop_dragging();
                }
            }

            if (const auto* mouseButtonReleased = event->getIf<sf::Event::MouseMoved>()) {
                if (is_dragging) need_render = true;
                dragging({ mouse.x, mouse.y });
            }
        }

        if (need_render) {
            // Configure CUDA kernel execution grid and block dimensions
            dim3 dimBlock(256, 256);
            dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

            // Launch Mandelbrot CUDA kernel
            mandelbrot <<<dimBlock, dimGrid>>> (d_pixels, width, height, zoom_x, zoom_y, x_offset, y_offset, d_palette, paletteSize, max_iterations);
            cudaDeviceSynchronize();


            // Copy rendered pixels back to host
            cudaMemcpy(pixels, d_pixels, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            sf::Image image1({ width, height }, pixels);
            image = image1;
            need_render = false;


            // Apply 4x SSAA
            compressed = ANTIALIASING_SSAA4(image, 4);

            window.clear();

            sf::Texture tt(compressed);
            sf::Sprite sprite(tt);

            sprite.setPosition({ 0, 0 });

            window.draw(sprite, sprite.getTransform());

            auto time = timer.restart();
            std::cout << "Mandelbrot set was drew in: " << time.asMilliseconds() << std::endl;


            // Display the rendered frame
            window.display();
        }
    }

    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_palette);

    return 0;
}