#pragma once
#include "fractals/mandelbrot.cuh"
#include "fractals/julia.cuh"
#include "CUDA_ComputationFunctions.cuh"
#include "benchmark.cuh"
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <cuda_runtime.h>
#include <random>
#include <thread>
#include <atomic>

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
    DeepSpace,
    Physchodelic,
    IceCave,
    AccretionDisk,
    ElectricNebula,
    Random
};

enum class render_state {
	good,
    best
};

struct timelapse_propertier {
    float zx = 0, zy = 0;
    float velocityX = 0, velocityY = 0;
};

struct render_target {
    render_target(unsigned int xst, unsigned int yst,
                  unsigned int xend, unsigned int yend) :
        x_start(xst), y_start(yst), x_end(xend), y_end(yend) {}
    unsigned int x_start;
    unsigned int y_start;
    unsigned int x_end;
    unsigned int y_end;
};

struct RenderGuard {
    std::atomic<bool>& flag;
    RenderGuard(std::atomic<bool>& f) : flag(f) {}
    ~RenderGuard() {
        flag.store(false);
    }
};

template <typename Derived>
class FractalBase : public sf::Transformable, public sf::Drawable {
protected:

    // THE MOST IMPORTANT VALUE
    bool isCudaAvailable = false;

    // CPU fallback properties
    std::vector<std::thread> threads;
    std::vector<render_target> render_targets;
    // 0 means thread is working
    // 1 means thread finished
    // 2 means thread is requested to stop
    std::vector<std::atomic<unsigned char>> thread_stop_flags;
    unsigned int max_threads = 0;
    std::atomic<bool> g_isCpuRendering = false;
    std::atomic<bool> allWorkDone = false;


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
    float maxComputationF, maxComputationD;

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
    unsigned int basic_width;
    unsigned int basic_height;
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

    /*
     * @returns mouse currently dragging state
     * */
    bool get_is_dragging();

    double get_x_offset();
    double get_y_offset();
    double get_zoom_x();
    double get_zoom_y();
    double get_zoom_scale();
    double get_hardness_coeff();

    void setMaxComputation(float Gflops, float GDflops);
    void setPallete(std::string name);
    Palletes getPallete();
    sf::Vector2i get_resolution() const;

    void SetDegreesOffsetForHSV(int degrees);
    sf::Texture getTexture();
    sf::Sprite get_sprite_rect();

    /*
    @brief sets max_iterations to new given number
    @param max_iters
    */
	void set_max_iters(unsigned int max_iters);

    void set_resolution(sf::Vector2i target_resolution);

    void post_processing();

    void reset();

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