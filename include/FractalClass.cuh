#pragma once
#include "fractals/mandelbrot.cuh"
#include "fractals/julia.cuh"
#include "CUDA_ComputationFunctions.cuh"
#include "benchmark.cuh"
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <cuda_runtime.h>
#include <random>

namespace fractals {
	struct mandelbrot{};
    struct julia{};
    struct burning_ship{};
};

enum class Palletes {
    HSV,
    BlackOWhite,
    Basic,
    Fire,
    Pastel,
    OscillatingGrayscale,
    Interpolated,
    CyclicHSV,
    FractalPattern,
    PerlinNoise,
    Water,
    Sunset,
    MultiDimensional,
    DeepSpace,
    Physchodelic,
    IceCave,
    AccretionDisk,
    ElectricNebula
};

extern bool running_other_core;

enum class render_state {
	good,
    best
};

struct timelapse_propertier {
    float zx = 0, zy = 0;
    float velocityX = 0, velocityY = 0;
};

template <typename Derived>
class FractalBase : public sf::Transformable, public sf::Drawable {
protected:
    // Fractal properties
    unsigned int max_iterations;
    double basic_zoom_x;
    double basic_zoom_y;
    double zoom_x;
    double zoom_y;
    double x_offset;
    double y_offset;
    double zoom_factor;
    double zoom_speed;
    double zoom_scale;
    float hardness_coeff = 0.f;
    Palletes curr_pallete = Palletes::HSV;
    unsigned int degrees_offsetForHSV = 0;
    unsigned int drawen_iterations = 0;
    float maxComputation;

    // Double precision stop flag
    int* d_doubleCancelFlag;
    int* h_doubleCancelFlag;

    // Dragging properties
    sf::Vector2i drag_start_pos;
    bool is_dragging = false;

    // Palette properties
    sf::Color* d_palette;
    std::vector<sf::Color> palette;
    int paletteSize;

    // Pixel buffers
    unsigned char* d_pixels;
    size_t ssaa_buffer_size = 0;
    unsigned char* pixels;
    size_t ssaa_limit_dst = 0;
    unsigned char* ssaa_buffer;
    unsigned char* compressed;

    // Rendering properties
    bool is_paused = false;
    render_state state = render_state::good;
    bool antialiasing = true;
    unsigned char counter = 0;

    // CUDA properties
    dim3 dimGrid;
    dim3 dimBlock;
    bool* stopFlagDevice;
    unsigned int width;
    unsigned int height;
    cudaStream_t stream;
    cudaStream_t dataStream;
    cudaEvent_t start_rendering, stop_rendering;
    unsigned int* d_total_iterations;
    unsigned int* h_total_iterations;

    // SFML properties
    sf::Image image;
    sf::Texture texture;
    sf::Sprite sprite;

    // Iteration lines
    std::vector<sf::Vertex> iterationpoints;
    sf::VertexBuffer iterationline;

    // Timelapse Properties
    std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<float> disX, disY;
    std::uniform_real_distribution<float> disVelX, disVelY;
    timelapse_propertier timelapse;
    sf::Clock clock;
    float velocityDiffX = 0, velocityDiffY = 0;
    float diffX = 0, diffY = 0;
    float initialVelocityX = 0, initialVelocityY = 0;
    float targetVelocityX = 0;
    float targetVelocityY = 0;
    float targetSpeed = 0.5f;
    unsigned int directionChangeInterval = 2000;
    float boundaryMargin = 0.1f;
    bool smoothing = false;


public:
    FractalBase();
    ~FractalBase();

    /*@returns amount of max_iterations*/
    unsigned int get_max_iters();

    /*@returns mouse currently dragging state*/
    bool get_is_dragging();

    double get_x_offset();
    double get_y_offset();
    double get_zoom_x();
    double get_zoom_y();
    double get_zoom_scale();
    double get_hardness_coeff();

    void setMaxComputation(float Gflops);
    void setPallete(std::string name);
    Palletes getPallete();
    void SetDegreesOffsetForHSV(int degrees);

    /*@brief sets max_iterations to new given number
    **@param max_iters
    */
	void set_max_iters(unsigned int max_iters);

    void post_processing();

    /**
     * @brief CUDA kernel function to calculate and render the fractal given template.
     * This kernel is executed in parallel by multiple threads on the GPU.
     * Each thread calculates the color of a single pixel based on its position
     * and the fractal rendering algorithm
     */
    void render(render_state quality);

    void render(
        render_state quality,
        double mouse_x, double mouse_y
    );

    void drawIterationLines(sf::Vector2i mouse_pos);

    /**
     * @brief Draws the rendered fractal image onto the SFML render target.
     * @param target The SFML render target (such as a window) where the image is drawn.
     * @param states SFML render states used to control how the drawing is done.
     */
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

    /**
     * @brief Handles zoom functionality based on mouse wheel input.
     * @param wheel_delta The direction and magnitude of the mouse wheel scroll.
     * @param mouse_pos The current mouse position in window coordinates.
     */
    void handleZoom(double wheel_delta, const sf::Vector2i mouse_pos);

    /**
     * @brief Starts the dragging operation when the mouse button is pressed.
     * @param mouse_pos The current mouse position.
     */
    void start_dragging(sf::Vector2i mouse_pos);

    /**
     * @brief Handles the dragging operation while the mouse is moved and the button is held down.
     * @param mouse_pos The current mouse position.
     */
    void dragging(sf::Vector2i mouse_pos);

    /**
     * @brief Stops the dragging operation when the mouse button is released.
     */
    void stop_dragging();

    /*@brief moves fractal by given coords
    * @param offset: x, y
    */
    void move_fractal(sf::Vector2i offset);

    void start_timelapse();

	void update_timelapse();

	void stop_timelapse();
};