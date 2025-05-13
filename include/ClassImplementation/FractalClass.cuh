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
#include <nvrtc.h>
#include <cuda.h>
#include <future>

/// Namespace to define structs representing different fractal types.
/// These are used as template parameters for `FractalBase`.
namespace fractals {
    struct mandelbrot{};
    struct julia{};
    struct burning_ship{}; // Not currently implemented in the provided code.

};

/// Enum for available color palettes.
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

/// Enum for rendering quality states.
/// 'good' is typically faster for interactive use, 'best' uses higher
/// resolution and anti-aliasing for final output.
enum class render_state {
    good,
    best
};

/// Enum for the CUDA computation context.
/// 'CUDA' refers to the CUDA Runtime API.
/// 'NVRTC' refers to using NVRTC for runtime compilation, implying the CUDA Driver API.
enum class context_type {
    CUDA,
    NVRTC
};

/// Struct to hold properties for the Julia set timelapse animation.
struct timelapse_propertier {
    float zx = 0, zy = 0; // Current complex parameter c for Julia set
    float velocityX = 0, velocityY = 0; // Velocity of the parameter c
};

/// Struct to define a rectangular region for rendering, typically used for CPU multithreading.
struct render_target {
    render_target(unsigned int xst, unsigned int yst,
                  unsigned int xend, unsigned int yend) :
            x_start(xst), y_start(yst), x_end(xend), y_end(yend) {}
    unsigned int x_start;
    unsigned int y_start;
    unsigned int x_end;
    unsigned int y_end;
};

/// RAII guard to manage an atomic boolean flag, ensuring it's set to false on destruction.
/// Used for managing the `g_isCpuRendering` lock.
struct RenderGuard {
    std::atomic<bool>& flag;
    RenderGuard(std::atomic<bool>& f) : flag(f) {}
    ~RenderGuard() {
        flag.store(false);
    }
};
/// FIXME: `initialized_nvrtc` is a static global variable. This could lead to issues if multiple
/// `FractalBase` instances (or other parts of a larger application) try to manage NVRTC initialization independently.
/// Consider making NVRTC initialization a singleton or part of a global application setup.
static bool initialized_nvrtc;

/// Type alias for a function pointer that creates a color palette.
using PaletteCreator = std::function<std::vector<uint8_t>(unsigned int)>;

/// Struct to associate a palette creation function with its enum value.
struct PaletteInfo {
    PaletteCreator creator;
    Palletes enum_value;
};

/// Static map to convert palette names (strings) to their corresponding enum values.
static const std::unordered_map<std::string, Palletes> s_palette_name_to_enum = {
        { "HSV", Palletes::HSV },
        { "Basic", Palletes::Basic },
        { "BlackOWhite", Palletes::BlackOWhite },
        { "OscillatingGrayscale", Palletes::OscillatingGrayscale },
        { "Interpolated", Palletes::Interpolated },
        { "Pastel", Palletes::Pastel },
        { "CyclicHSV", Palletes::CyclicHSV },
        { "Fire", Palletes::Fire },
        { "FractalPattern", Palletes::FractalPattern },
        { "PerlinNoise", Palletes::PerlinNoise },
        { "Water", Palletes::Water },
        { "Sunset", Palletes::Sunset },
        { "DeepSpace", Palletes::DeepSpace },
        { "Physchodelic", Palletes::Physchodelic },
        { "IceCave", Palletes::IceCave },
        { "AccretionDisk", Palletes::AccretionDisk },
        { "ElectricNebula", Palletes::ElectricNebula },
        { "Random", Palletes::Random }
};

/// Base class for fractal rendering, providing common functionality.
/// It is templated on a `Derived` type (e.g., `fractals::mandelbrot`) to allow
/// for specific fractal logic.
template <typename Derived>
class FractalBase : public sf::Transformable, public sf::Drawable {
protected:

    // Custom Formula Properties (NVRTC related)
    std::thread compile_thread;          // Thread for asynchronous NVRTC compilation
    std::string compute_capability;      // GPU compute capability string (e.g., "compute_75")
    std::future<std::string> current_compile_future; // Future to get compilation result/log
    std::atomic<unsigned int> progress_compiling_percentage = 0; // Compilation progress
    std::string log_buffer;              // Buffer for NVRTC compilation log
    context_type context = context_type::CUDA; // Current CUDA context (Runtime or NVRTC/Driver)
    bool custom_formula = false;         // Flag indicating if a custom formula is active
    std::string kernel_code;             // Stores the custom kernel code string (not explicitly used in provided snippets)
    CUcontext ctx;                       // CUDA Driver API context
    CUdevice device;                     // CUDA Driver API device
    CUmodule module;                     // CUDA Driver API module (for NVRTC compiled code)
    std::atomic<bool> is_compiling = false; // Flag indicating if compilation is in progress
    bool module_loaded = false;          // Flag indicating if an NVRTC module is loaded
    bool created_context;                // Flag indicating if a CUDA Driver API context was created by this instance
    CUfunction kernelFloat;              // CUDA Driver API function pointer (float precision custom kernel)
    CUfunction kernelDouble;             // CUDA Driver API function pointer (double precision custom kernel)
    CUfunction kernelAntialiasing;       // CUDA Driver API function pointer (anti-aliasing kernel)
    CUdeviceptr cu_d_total_iterations;   // Device pointer for total iterations (Driver API)
    CUdeviceptr cu_d_pixels;             // Device pointer for pixel data (Driver API)
    CUdeviceptr cu_palette;              // Device pointer for palette (Driver API)
    CUdeviceptr CUssaa_buffer;           // Device pointer for SSAA buffer (Driver API)
    CUstream CUss;                       // CUDA stream for NVRTC rendering (Driver API)
    CUstream CUssData;                   // CUDA stream for NVRTC data transfers (Driver API)


    /// Flag indicating if a CUDA-capable GPU is available.
    bool isCudaAvailable = false;

    // CPU fallback properties
    std::vector<std::thread> threads;    // Worker threads for CPU rendering
    std::vector<render_target> render_targets; // Defines regions for each CPU thread to render
    /// Atomic flags to control and monitor CPU worker threads:
    /// 0: thread is working
    /// 1: thread finished
    /// 2: thread is requested to stop
    std::vector<std::atomic<unsigned char>> thread_stop_flags;
    unsigned int max_threads = 0;        // Max CPU threads to use (initialized from hardware_concurrency)
    /// Global atomic flag to act as a mutex, ensuring only one CPU rendering operation runs at a time across all FractalBase instances.
    std::atomic<bool> g_isCpuRendering = false;


    // Fractal properties
    unsigned int max_iterations;
    double basic_zoom_x;                 // Base zoom level (before zoom_factor)
    double basic_zoom_y;
    double zoom_x;                       // Current effective zoom (basic_zoom * zoom_factor)
    double zoom_y;
    double x_offset;                     // Offset in the complex plane
    double y_offset;
    double zoom_factor;                  // Accumulative zoom multiplier from user input
    double zoom_speed;                   // Sensitivity of zooming
    double zoom_scale;                   // Scale factor for rendering (1.0 for normal, 2.0 for SSAA)
    double zreal, zimag;                 // Stores the 'c' constant for Julia sets
    float hardness_coeff = 0.f;          // Average iterations per pixel, measures computational difficulty
    Palletes curr_pallete = Palletes::HSV; // Current active color palette
    unsigned int degrees_offsetForHSV = 0; // Hue offset for HSV-based palettes
    unsigned int drawen_iterations = 0;  // Number of points to draw for the iteration path
    float maxComputationF, maxComputationD; // Estimated GPU GFLOPS/GDFLOPS for performance scaling

    // Dragging properties
    sf::Vector2i drag_start_pos;
    bool is_dragging = false;

    // Palette properties
    Color* d_palette;                    // Device pointer for the color palette (CUDA Runtime)
    std::vector<Color> palette;          // Host-side color palette
    int paletteSize;

    // Pixel buffers
    unsigned char* d_pixels;             // Device pointer for pixel data (CUDA Runtime, main render target)
    unsigned char* pixels;               // Host (pinned) memory for pixel data (transfer from d_pixels or for CPU rendering)
    unsigned char* ssaa_buffer;          // Device pointer for anti-aliasing buffer (CUDA Runtime)
    unsigned char* compressed;           // Host (pinned) memory for anti-aliased pixel data

    // Rendering properties
    render_state state = render_state::good; // Current rendering quality state
    bool antialiasing = true;                // Flag to enable/disable anti-aliasing

    // CUDA properties (Runtime API)
    dim3 dimGrid;                        // Grid dimensions for kernel launch
    dim3 dimBlock;                       // Block dimensions for kernel launch
    unsigned int basic_width;            // Base rendering width (before SSAA scaling)
    unsigned int basic_height;           // Base rendering height
    unsigned int width;                  // Current rendering width (can be basic_width or 2*basic_width for SSAA)
    unsigned int height;                 // Current rendering height
    cudaStream_t stream;                 // Main CUDA stream for rendering (Runtime API)
    cudaStream_t dataStream;             // Separate CUDA stream for data transfers (Runtime API)
    unsigned int* d_total_iterations;    // Device pointer for total iterations (CUDA Runtime)
    unsigned int* h_total_iterations;    // Host (pinned) memory for total iterations

    // SFML properties
    sf::Image image;                     // SFML Image for CPU-side pixel manipulation and texture loading
    sf::Texture texture;                 // SFML Texture for GPU-accelerated drawing
    sf::Sprite sprite;                   // SFML Sprite to draw the texture

    // Iteration lines
    std::vector<sf::Vertex> iterationpoints; // Stores vertices for the iteration path
    sf::VertexBuffer iterationline;          // SFML VertexBuffer for efficient drawing of the iteration path

    // Timelapse Properties
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> disX, disY; // Distributions for random Julia parameters
    std::uniform_real_distribution<float> disVelX, disVelY; // Distributions for random velocities
    timelapse_propertier timelapse;      // Holds current state for timelapse animation
    sf::Clock clock;                     // Clock for timing timelapse updates


public:
    FractalBase();
    ~FractalBase();

    unsigned int get_max_iters();
    bool get_is_dragging();
    bool get_bool_custom_formula();
    double get_x_offset();
    double get_y_offset();
    double get_zoom_x();
    double get_zoom_y();
    double get_zoom_scale();
    double get_hardness_coeff();
    unsigned int get_compiling_percentage();
    bool get_is_compiling() {return is_compiling;}
    bool get_isCudaAvailable() {return isCudaAvailable;}

    void setMaxComputation(float Gflops, float GDflops);

    void setPallete(std::string name);
    Palletes getPallete();

    sf::Vector2i get_resolution() const;

    void SetDegreesOffsetForHSV(int degrees);

    sf::Texture getTexture();

    void set_max_iters(unsigned int max_iters);
    void set_context(context_type ctx);
    void set_resolution(sf::Vector2i target_resolution);
    void set_grid(dim3 block);

    void post_processing();
    void reset();

    std::shared_future<std::string> set_custom_formula(const std::string& formula);
    context_type get_context();

    void render(render_state quality);
    void render(
            render_state quality,
            double mouse_x, double mouse_y
    );
    void drawIterationLines(sf::Vector2i mouse_pos);
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void handleZoom(double wheel_delta, const sf::Vector2i mouse_pos);
    void start_dragging(sf::Vector2i mouse_pos);
    void dragging(sf::Vector2i mouse_pos);
    void stop_dragging();
    void move_fractal(sf::Vector2i offset);
    void start_timelapse();
    void update_timelapse();
    void stop_timelapse();
};