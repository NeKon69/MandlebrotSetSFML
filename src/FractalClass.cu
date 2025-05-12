/// Includes necessary headers for fractal rendering (Julia, Mandelbrot),
/// custom formula handling (NVRTC), CUDA processing utilities,
/// Julia set timelapse functionality, fractal interaction logic (zoom, pan),
/// color palette management, iteration path drawing, and TGUI backend integration.
#include "ClassImplementation/Fractals/JuliaRendering.cuh"
#include "ClassImplementation/Fractals/MandelbrotRendering.cuh"
#include "ClassImplementation/CustomFormulaHandling.h"
#include "ClassImplementation/Processing.cu"
#include "ClassImplementation/JuliaTimelapse.h"
#include "ClassImplementation/FractalInteraction.h"
#include "ClassImplementation/PaletteHandler.h"
#include "ClassImplementation/IterationPath.h"
#include <TGUI/Backend/SFML-Graphics.hpp>

/// Template class FractalBase implementation.
/// Provides common functionality for different fractal types (Mandelbrot, Julia).
/// Manages fractal parameters, rendering state, memory, and user interaction.
template <typename Derived>
FractalBase<Derived>::FractalBase()
        :
        /// Initializes flags for potential thread management
        thread_stop_flags(std::thread::hardware_concurrency() * 100),
        /// Initializes basic and current fractal parameters like iterations, zoom, offset, and speed.
        max_iterations(MAX_ITERATIONS), basic_zoom_x(BASIC_ZOOM_X), basic_zoom_y(BASIC_ZOOM_Y),
        zoom_x(basic_zoom_x), zoom_y(basic_zoom_y),
        x_offset(BASIC_X_OFFSET), y_offset(BASIC_Y_OFFSET),
        zoom_factor(BASIC_ZOOM_FACTOR), zoom_speed(BASIC_ZOOM_SPEED),
        zoom_scale(BASIC_ZOOM_SCALE),  maxComputationF(BASIC_MAX_COMPUTATION_F), maxComputationD(BASIC_MAX_COMPUTATION_D),
        /// Initializes basic and current resolution.
        basic_width(BASIC_WIDTH), basic_height(BASIC_HEIGHT),
        width(basic_width), height(basic_height),
        /// Initializes SFML sprite and texture for rendering, and primitive type for iteration lines.
        sprite(texture), iterationline(sf::PrimitiveType::LineStrip),
        /// Sets up random number generators for timelapse.
        gen(rd()),
        disX(-2.f, 2.f), disY(-1.5f, 1.5f),
        disVelX(-0.13f, 0.13f),disVelY(-0.1f, 0.1f)
{
    initialized_nvrtc = false;
    created_context = false;

    /// Checks for CUDA device availability. If no device is found,
    /// it sets a flag indicating CUDA is not available, forcing the use
    /// of CPU rendering (handled elsewhere, but noted here).
    /// If CUDA is available, it gets device properties to set the
    /// compute capability string for NVRTC compilation.
    /// Creates a default color palette (HSV).
    palette = createHSVPalette(BASIC_PALETTE_SIZE);
    paletteSize = BASIC_PALETTE_SIZE;
    /// Resizes the iteration points vector, used for drawing iteration paths.
    iterationpoints.resize(max_iterations);

    isCudaAvailable = true;
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if(numDevices == 0) {
        std::cout << "IMPORTANT NO AVAILABLE CUDA DEVICES FOUND" << std::endl;
        std::cout << "Forcing to use CPU rendering" << std::endl;
        std::cout << "Please make sure you have CUDA installed and your GPU supports it" << std::endl;
        isCudaAvailable = false;
        h_total_iterations = static_cast<unsigned int *>(malloc(sizeof(unsigned int)));
        compressed = static_cast<unsigned char *>(malloc(basic_width * basic_height * 4 * sizeof(unsigned char)));
        pixels = static_cast<unsigned char *>(malloc(basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)));
    }
    else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        compute_capability = "--gpu-architecture=compute_" + std::to_string(deviceProp.major) + std::to_string(deviceProp.minor);
        /// Allocates necessary memory on both host and device (GPU) for image data.
        ALLOCATE_ALL_IMAGE_MEMORY();
        /// Allocates necessary memory on both host and device (GPU) for non-image data (e.g., iteration counts).
        ALLOCATE_ALL_NON_IMAGE_MEMORY();
    }
}

/// Destructor for the FractalBase class.
/// Ensures proper cleanup of GPU resources and allocated memory.
template <typename Derived>
FractalBase<Derived>::~FractalBase() {
    if(!isCudaAvailable) {
        free(compressed);
        free(pixels);
        free(h_total_iterations);
    }
    else {
        /// Frees all allocated memory on both host and device (GPU).
        FREE_ALL_IMAGE_MEMORY();
        FREE_ALL_NON_IMAGE_MEMORY();
        /// Unloads the NVRTC module if it was loaded.
        if (module_loaded) CU_SAFE_CALL(cuModuleUnload(module));
        /// Destroys the CUDA context if it was created, using a macro
        /// to handle context switching if necessary.
        MAKE_CURR_CONTEXT_OPERATION(cudaFree(nullptr), cuCtxDestroy(ctx), context);
    }
}

template <typename Derived>
unsigned int FractalBase<Derived>::get_max_iters() { return max_iterations; }

template <typename Derived>
bool FractalBase<Derived>::get_is_dragging() { return is_dragging; }

template <typename Derived>
void FractalBase<Derived>::set_max_iters(unsigned int max_iters) { max_iterations = max_iters; }

template <typename Derived>
double FractalBase<Derived>::get_x_offset() { return x_offset; }

template <typename Derived>
double FractalBase<Derived>::get_y_offset() { return y_offset; }

template <typename Derived>
double FractalBase<Derived>::get_zoom_x() { return zoom_x; }

template <typename Derived>
double FractalBase<Derived>::get_zoom_y() { return zoom_y; }

template <typename Derived>
double FractalBase<Derived>::get_zoom_scale() { return zoom_scale; }

template <typename Derived>
double FractalBase<Derived>::get_hardness_coeff() { return hardness_coeff; }

template <typename Derived>
sf::Texture FractalBase<Derived>::getTexture() { return texture; }

/// Sets the maximum computation capacity based on measured GPU performance (GFLOPS/GDFLOPS).
template <typename Derived>
void FractalBase<Derived>::setMaxComputation(float Gflops, float GDflops) { maxComputationF = 50.0f / 90 * Gflops; maxComputationD = 50.0f / 90 * GDflops; }

template <typename Derived>
Palletes FractalBase<Derived>::getPallete() { return curr_pallete; }

template <typename Derived>
sf::Vector2i FractalBase<Derived>::get_resolution() const { return {int(basic_width), int(basic_height)}; }

template <typename Derived>
unsigned int FractalBase<Derived>::get_compiling_percentage() {
    if (context == context_type::NVRTC) {
        return progress_compiling_percentage;
    }
    else {
        return 0;
    }
}

/// Sets the resolution of the fractal image.
/// When the resolution changes, the zoom and offset are recalculated
/// to keep the center of the previous view the same, maintaining
/// the visual focus point.
/// Existing GPU memory is freed and reallocated for the new resolution.
template <typename Derived>
void FractalBase<Derived>::set_resolution(sf::Vector2i target_resolution) {
    unsigned int old_width = width, old_height = height;

    width = basic_width = target_resolution.x;
    height = basic_height = target_resolution.y;

    double center_x = x_offset + (old_width / (zoom_x * zoom_scale)) / 2.0;
    double center_y = y_offset + (old_height / (zoom_y * zoom_scale)) / 2.0;

    zoom_x = basic_zoom_x * zoom_factor;
    zoom_y = basic_zoom_y * zoom_factor;

    x_offset = center_x - (width / (zoom_x * zoom_scale)) / 2.0;
    y_offset = center_y - (height / (zoom_y * zoom_scale)) / 2.0;

    FREE_ALL_IMAGE_MEMORY();
    ALLOCATE_ALL_IMAGE_MEMORY();
}


/// Resets the fractal view and parameters to their initial default state.
/// This includes resetting zoom, offset, iterations, and palette.
/// All existing GPU memory is freed and reallocated to ensure a clean state.
template <typename Derived>
void FractalBase<Derived>::reset() {
    // Free existing resources
    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();

    INIT_BASIC_VALUES;

    if (std::is_same<Derived, fractals::julia>::value) {
        x_offset = 2.5;
        palette = createHSVPalette(BASIC_PALETTE_SIZE);
        paletteSize = BASIC_PALETTE_SIZE;
    }
    else {
        palette = createHSVPalette(BASIC_PALETTE_SIZE);
        paletteSize = BASIC_PALETTE_SIZE;
    }

    if(context == context_type::NVRTC) cuCtxSetCurrent(ctx);
    ALLOCATE_ALL_IMAGE_MEMORY();
    ALLOCATE_ALL_NON_IMAGE_MEMORY();

    iterationpoints.resize(max_iterations);
}

// that code served me good in the past, however it's being replaced with better version with atomic operations
// nevermind, atomic sucks!!! async is the way to go
// UwU sooooo saaaad UwU
//template <typename Derived>
//void FractalBase<Derived>::checkEventAndSetFlag(cudaEvent_t event) {
//    while (cudaEventQuery(event) == cudaErrorNotReady) {
//        std::this_thread::sleep_for(std::chrono::milliseconds(1));
//    }
//    bool flag = false;
//    running_other_core = false;
//    cudaMemcpy(stopFlagDevice, &flag, sizeof(bool), cudaMemcpyHostToDevice);
//}

template <typename Derived>
context_type FractalBase<Derived>::get_context() { return context; }

template <typename Derived>
bool FractalBase<Derived>::get_bool_custom_formula() { return custom_formula; }

template class FractalBase<fractals::mandelbrot>;
template class FractalBase<fractals::julia>;