#pragma once
#include "fractals/mandelbrot.cuh"
#include "fractals/julia.cuh"
#include "CUDA_ComputationFunctions.cuh"
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

namespace fractals {
	struct mandelbrot{};
    struct julia{};
    struct burning_ship{};
};

extern bool running_other_core;

enum class render_state {
	bad,
	good,
    best
};

template <typename Derived>
class FractalBase : public sf::Transformable, public sf::Drawable {
protected:
    unsigned int max_iterations;

    double basic_zoom_x;
    double basic_zoom_y;

    double zoom_x;
    double zoom_y;

    double x_offset;
    double y_offset;

    double zoom_factor;
    double zoom_speed;

    sf::Vector2i drag_start_pos;
    bool is_dragging = false;

    sf::Color* d_palette;

    int paletteSize;

    unsigned char* d_pixels;
    size_t ssaa_buffer_size = 0;

    unsigned char* pixels;
    size_t ssaa_limit_dst = 0;

	unsigned char* ssaa_buffer;

    unsigned char* compressed;

	bool is_paused = false;

	render_state state = render_state::good;
	bool antialiasing = true;

    bool* stopFlagDevice;
    std::atomic<bool> stopFlagCpu;

    double zoom_scale;

    unsigned int width;
    unsigned int height;

    cudaStream_t stream;
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


    /*@brief sets max_iterations to new given number
    **@param max_iters
    */
	void set_max_iters(unsigned int max_iters);



    void checkEventAndSetFlag(cudaEvent_t event);

    /**
     * @brief CUDA kernel function to calculate and render the fractal given template.
     * This kernel is executed in parallel by multiple threads on the GPU.
     * Each thread calculates the color of a single pixel based on its position
     * and the fractal rendering algorithm.
     *
     * @param width Image width.
     * @param height Image height.
     */
    void render(render_state quality);

    void render(
        render_state quality,
        double mouse_x, double mouse_y
    );

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
    void handleZoom(float wheel_delta, const sf::Vector2i mouse_pos);

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
};