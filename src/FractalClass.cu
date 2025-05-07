#include "ClassImplementation/Fractals/JuliaRendering.cuh"
#include "ClassImplementation/Fractals/MandelbrotRendering.cuh"
#include "ClassImplementation/CustomFormulaHandling.h"
#include "ClassImplementation/Processing.cu"
#include "ClassImplementation/JuliaTimelapse.h"
#include "ClassImplementation/FractalInteraction.h"
#include "ClassImplementation/PaletteHandler.h"
#include "ClassImplementation/IterationPath.h"
#include <TGUI/Backend/SFML-Graphics.hpp>

template <typename Derived>
FractalBase<Derived>::FractalBase()
    :
    thread_stop_flags(std::thread::hardware_concurrency() * 100),
    max_iterations(300), basic_zoom_x(240.0), basic_zoom_y(240.0),
    zoom_x(basic_zoom_x), zoom_y(basic_zoom_y),
    x_offset(2.25), y_offset(1.25),
    zoom_factor(1.0), zoom_speed(0.1),
    zoom_scale(1.0),  maxComputationF(50.f), maxComputationD(50.f),
    basic_width(800), basic_height(600),
    width(800), height(600),
    sprite(texture), iterationline(sf::PrimitiveType::LineStrip),
    gen(rd()),
    disX(-2.f, 2.f), disY(-1.5f, 1.5f),
    disVelX(-0.13f, 0.13f),disVelY(-0.1f, 0.1f)
{
    initialized_nvrtc = false;
    created_context = false;

    isCudaAvailable = true;
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if(numDevices == 0) {
        std::cout << "IMPORTANT NO AVAILABLE CUDA DEVICES FOUND" << std::endl;
        std::cout << "Forcing to use CPU rendering" << std::endl;
        std::cout << "Please make sure you have CUDA installed and your GPU supports it" << std::endl;
        isCudaAvailable = false;
    }
    isCudaAvailable = false;
    palette = createHSVPalette(20000);
    paletteSize = 20000;


    ALLOCATE_ALL_IMAGE_MEMORY();
    ALLOCATE_ALL_NON_IMAGE_MEMORY();

    iterationpoints.resize(max_iterations);

}

template <typename Derived>
FractalBase<Derived>::~FractalBase() {
    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();
    if (module_loaded) CU_SAFE_CALL(cuModuleUnload(module));
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(nullptr), cuCtxDestroy(ctx), context);
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

template<typename Derived>
sf::Sprite FractalBase<Derived>::get_sprite_rect() { return sprite; }

template <typename Derived>
void FractalBase<Derived>::setMaxComputation(float Gflops, float GDflops) { maxComputationF = 50.0f / 90 * Gflops; maxComputationD = 50.0f / 90 * GDflops; }

template <typename Derived>
Palletes FractalBase<Derived>::getPallete() { return curr_pallete; }

template <typename Derived>
sf::Vector2i FractalBase<Derived>::get_resolution() const { return {int(basic_width), int(basic_height)}; }

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


// To surely not forget anything lets make sure to delete everything and reallocate
template <typename Derived>
void FractalBase<Derived>::reset() {
    // Free existing resources
    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();

    INIT_BASIC_VALUES;

    if (std::is_same<Derived, fractals::julia>::value) {
        x_offset = 2.5;
        palette = createHSVPalette(20000);
        paletteSize = 20000;
    }
    else {
        palette = createHSVPalette(20000);
        paletteSize = 20000;
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