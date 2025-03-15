#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <complex>
#include <cmath>
#include <vector_alias.hpp>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <thread>



unsigned int max_iterations = 500;

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

unsigned char* pixels = new unsigned char[width * height * 4];
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


__global__
void mandelbrot(unsigned char* pixels, int width, int height, double zoom_x, double zoom_y, double x_offset, double y_offset, sf::Color* d_palette, int paletteSize, double maxIterations) {
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
            double max_iters = maxIterations;
            current_iteration = current_iteration + 1 - dev_log2(dev_log2(dev_abs(dev_sqrt(complex_abs2(z_real, z_imag)))));
            double gradient = Gradient(current_iteration, max_iters);
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

void handleZoom(float wheel_delta, const sf::Vector2i mouse_pos) {
    double old_zoom_x = zoom_x;
    double old_zoom_y = zoom_y;
    double old_x_offset = x_offset;
    double old_y_offset = y_offset;

    double zoom_change = 1.0 + wheel_delta * zoom_speed;
    zoom_factor *= zoom_change;

    zoom_factor = std::max(std::min(zoom_factor, 1000000000000.0), 0.01);

    zoom_x = basic_zoom_x * zoom_factor;
    zoom_y = basic_zoom_y * zoom_factor;

    double image_mouse_x = mouse_pos.x * (1600.0 / 800.0);
    double image_mouse_y = mouse_pos.y * (1200.0 / 600.0);

    x_offset = old_x_offset + (image_mouse_x / zoom_x - image_mouse_x / old_zoom_x);
    y_offset = old_y_offset + (image_mouse_y / zoom_y - image_mouse_y / old_zoom_y);

    std::cout << "Zoom Factor: " << zoom_factor << ", Offset: (" << x_offset << ", " << y_offset << ")" << std::endl;
}

void start_dragging(sf::Vector2i mouse_pos) {
    is_dragging = true;
    drag_start_pos = mouse_pos;
}

void dragging(sf::Vector2i mouse_pos) {
    if (!is_dragging) return;

    sf::Vector2i delta_pos = mouse_pos - drag_start_pos;

    double delta_real = static_cast<double>(delta_pos.x) / zoom_x;
    double delta_imag = static_cast<double>(delta_pos.y) / zoom_y;

    x_offset += delta_real;
    y_offset += delta_imag;

    drag_start_pos = mouse_pos;
}

void stop_dragging() {
    is_dragging = false;
}

int main() {
    bool need_render = true;
    sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "Mandelbrot");
    sf::Image image({ 1600, 1200 }, sf::Color::Black);
    sf::Image compressed({ 800, 600 }, sf::Color::Black);
    sf::Clock timer;

    double max_iters = max_iterations;

    sf::Vector2i mouse;

    // Palette transfer to device
    sf::Color* d_palette;
    int paletteSize = palette.size();
    cudaMalloc(&d_palette, paletteSize * sizeof(sf::Color));
    cudaMemcpy(d_palette, palette.data(), paletteSize * sizeof(sf::Color), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pixels, width * height * 4 * sizeof(unsigned char));

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
            dim3 dimBlock(128, 128);
            dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
            mandelbrot <<<dimBlock, dimGrid>>> (d_pixels, width, height, zoom_x, zoom_y, x_offset, y_offset, d_palette, paletteSize, max_iterations);
            cudaDeviceSynchronize();
            cudaMemcpy(pixels, d_pixels, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            sf::Image image1({ width, height }, pixels);
            image = image1;
            need_render = false;

            compressed = ANTIALIASING_SSAA4(image, 4);

            window.clear();

            sf::Texture tt(compressed);
            sf::Sprite sprite(tt);

            sprite.setPosition({ 0, 0 });

            window.draw(sprite, sprite.getTransform());

            auto time = timer.restart();
            std::cout << "Mandelbrot set was drew in: " << time.asMilliseconds() << std::endl;

            window.display();
        }
    }

    cudaFree(d_pixels);
    cudaFree(d_palette);

    return 0;
}