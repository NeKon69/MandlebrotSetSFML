#include "FractalClass.cuh"
#include <cuda_runtime.h>
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <iostream>
#include <thread>
#include <random>
#include <functional>
#include <fstream>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      throw std::runtime_error("ERR");                                                     \
    }                                                             \
} while(0)

#define CU_SAFE_CALL(x)                                           \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      throw std::runtime_error(#x " failed with error " + std::string(msg));                                                     \
    }                                                             \
} while(0)

#define ALLOC_AND_COPY_TO_DEVICE_CU(cu_devPtr, hostVar, type, num)          \
  CU_SAFE_CALL(cuMemAlloc(&cu_devPtr, sizeof(type) * num));               \
  CU_SAFE_CALL(cuMemcpyHtoD(cu_devPtr, &hostVar, sizeof(type) * num));

#define CUDA_SAFE_CALL(x) \
  do {                    \
    cudaError_t result = x;                                         \
    if (result != cudaSuccess) {                                   \
      const char *msg =  cudaGetErrorName(result);                  \
      std::cerr << "\nerror: " #x " failed with error "             \
                << msg << '\n';                                     \
      throw std::runtime_error(#x " failed with error " + std::string(msg));                                                       \
    }                                                               \
  }while(0)


#define MAKE_CURR_CONTEXT_OPERATION(x, y, ctx)                      \
  do {                                                              \
    if (context == context_type::CUDA){                             \
      CUDA_SAFE_CALL(x);                                            \
    }                                                               \
    else{                                                           \
      CU_SAFE_CALL(y);                                              \
    }                                                               \
} while(0)


#define COPY_PALETTE_TO_DEVICE(host, d, cu, ctx)                                                       \
  do {                                                                                          \
    if (context == context_type::CUDA){                                                         \
      CUDA_SAFE_CALL(cudaMemcpy(d, host, sizeof(Color) * paletteSize, cudaMemcpyHostToDevice));    \
    }                                                                                           \
    else{                                                                                       \
      CU_SAFE_CALL(cuMemcpyHtoD(cu, host, sizeof(Color) * paletteSize));                            \
    }                                                                                           \
} while(0)

#define ALLOCATE_ALL_IMAGE_MEMORY()                                                                                                                                                                 \
  do{                                                                                                                                                                                               \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_pixels, basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)), cuMemAlloc(&cu_d_pixels, sizeof(unsigned char) * basic_width * 2 * basic_height * 2 * 4), context);                  \
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&pixels, basic_width * 2 * basic_height * 2 * 4 * sizeof(unsigned char)), cuMemHostAlloc((void**)&pixels, sizeof(unsigned char) * basic_width * 2 * basic_height * 2 * 4, 0), context);      \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&ssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char)), cuMemAlloc(&CUssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char)), context);                             \
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&compressed, basic_width * basic_height * 4 * sizeof(unsigned char)), cuMemHostAlloc((void**)&compressed, basic_width * basic_height * 4 * sizeof(unsigned char), 0), context);                                                                      \
  } while(0)

#define FREE_ALL_IMAGE_MEMORY()                                                                 \
  do {                                                                                          \
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_pixels), cuMemFree(cu_d_pixels), context);           \
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(ssaa_buffer), cuMemFree(CUssaa_buffer), context);      \
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(pixels), cuMemFreeHost(pixels), context);          \
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(compressed), cuMemFreeHost(compressed), context);  \
  } while(0)

#define ALLOCATE_ALL_NON_IMAGE_MEMORY() \
  do {                                    \
    unsigned int zero = 0; \
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_palette, palette.size() * sizeof(Color)), cuMemAlloc(&cu_palette, sizeof(Color) * paletteSize), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(Color), cudaMemcpyHostToDevice), cuMemcpyHtoD(cu_palette, palette.data(), sizeof(Color) * paletteSize), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamCreate(&stream), cuStreamCreate(&CUss, 0), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMalloc(&d_total_iterations, sizeof(unsigned int)), cuMemAlloc(&cu_d_total_iterations, sizeof(unsigned int)), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMemset(d_total_iterations, 0, sizeof(unsigned int)), cuMemcpyHtoD(cu_d_total_iterations, &zero, sizeof(unsigned int)), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaMallocHost(&h_total_iterations , sizeof(unsigned int)), cuMemHostAlloc((void**)&h_total_iterations, sizeof(unsigned int), 0), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamCreate(&dataStream), cuStreamCreate(&CUssData, 0), context);\
  } while(0)

#define FREE_ALL_NON_IMAGE_MEMORY() \
  do {                               \
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamDestroy(stream), cuStreamDestroy(CUss), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFreeHost(h_total_iterations), cuMemFreeHost(h_total_iterations), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaStreamDestroy(dataStream), cuStreamDestroy(CUssData), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_total_iterations), cuMemFree(cu_d_total_iterations), context);\
    MAKE_CURR_CONTEXT_OPERATION(cudaFree(d_palette), cuMemFree(cu_palette), context);\
  } while(0)


void cpu_render_mandelbrot(render_target target, unsigned char* pixels, unsigned int width, unsigned int height, double zoom_x, double zoom_y,
    double x_offset, double y_offset, Color* palette, unsigned int paletteSize,
    unsigned int max_iterations, unsigned int* total_iterations, std::atomic<unsigned char>& finish_flag
    )
{
    try {

        finish_flag.store(0);
        if (target.x_end > width) {
            target.x_end = width;
        }
        if (target.y_end > height) {
            target.y_end = height;
        }
        if (target.x_start < 0) {
            target.x_start = 0;
        }
        if (target.y_start < 0) {
            target.y_start = 0;
        }
        for(unsigned int y = target.y_start; y < target.y_end; ++y){
            for(unsigned int x = target.x_start; x < target.x_end; ++x){
                double zr = 0.0;
                double zi = 0.0;
                double cr = x / zoom_x - x_offset;
                double ci = y / zoom_y - y_offset;
                unsigned char r, g, b;
                float curr_iter = 0;
                while (curr_iter < max_iterations && zr * zr + zi * zi < 100.0) {
                    double tmp_zr = zr;
                    zr = zr * zr - zi * zi + cr;
                    zi = 2.0 * tmp_zr * zi + ci;

                    ++curr_iter;
                    if(finish_flag.load() == 2) {
                        finish_flag.store(1);
                        return;
                    }
                }
                if (curr_iter == max_iterations) {
                    r = g = b = 0;
                } else {
                    double smooth_iteration = curr_iter + 1.0f - log2f(log2f(sqrtf(zr * zr + zi * zi)));

                    const double cycle_scale_factor = 25.0f;
                    double virtual_pos = smooth_iteration * cycle_scale_factor;

                    double normalized_pingpong = fmodf(virtual_pos / static_cast<double>(paletteSize -1), 2.0f);
                    if (normalized_pingpong < 0.0f) {
                        normalized_pingpong += 2.0f;
                    }

                    double t_interp;
                    if (normalized_pingpong <= 1.0f) {
                        t_interp = normalized_pingpong;
                    } else {
                        t_interp = 2.0f - normalized_pingpong;
                    }

                    double float_index = t_interp * (paletteSize - 1);

                    int index1 = static_cast<int>(floorf(float_index));
                    int index2 = std::min(int(paletteSize - 1), index1 + 1);

                    index1 = std::max(0, index1);

                    double t_local = fmodf(float_index, 1.0f);
                    if (t_local < 0.0f) t_local += 1.0f;

                    Color color1 = getPaletteColor(index1, paletteSize, palette);
                    Color color2 = getPaletteColor(index2, paletteSize, palette);

                    float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
                    float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
                    float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

                    r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r_f)));
                    g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g_f)));
                    b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b_f)));
                }
                const unsigned int index = (y * width + x) * 4;
                pixels[index] = r;
                pixels[index + 1] = g;
                pixels[index + 2] = b;
                pixels[index + 3] = 255;
                *total_iterations += curr_iter;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR in cpu_render_mandelbrot thread (target y=" << target.y_start << "): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "ERROR in cpu_render_mandelbrot thread (target y=" << target.y_start << "): Unknown exception!" << std::endl;
    }
    finish_flag.store(1);
}


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
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(numDevices == 0) {
        std::cout << "IMPORTANT NO AVAILABLE CUDA DEVICES FOUND" << std::endl;
        std::cout << "Forcing to use CPU rendering" << std::endl;
        std::cout << "Please make sure you have CUDA installed and your GPU supports it" << std::endl;
        isCudaAvailable = false;
    }
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
void FractalBase<Derived>::setPallete(std::string name) {
    if (name == "HSV") {
		palette = createHSVPalette(20000, degrees_offsetForHSV);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
        curr_pallete = Palletes::HSV;
    }
    if (name == "Basic") {
		palette = BluePlusBlackWhitePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::Basic;
    }
    if (name == "BlackOWhite") {
		palette = CreateBlackOWhitePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::BlackOWhite;
    }
    if (name == "OscillatingGrayscale") {
		palette = CreateOscillatingGrayscalePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::OscillatingGrayscale;
    }
    if (name == "Interpolated") {
		palette = CreateInterpolatedPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::Interpolated;
    }
    if (name == "Pastel") {
        palette = CreatePastelPalette(20000);
        paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
        curr_pallete = Palletes::Pastel;
    }
    if (name == "CyclicHSV") {
		palette = CreateCyclicHSVPpalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::CyclicHSV;
    }
    if (name == "Fire") {
		palette = CreateFirePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::Fire;
    }
    if (name == "FractalPattern") {
		palette = CreateFractalPatternPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::FractalPattern;
    }
    if (name == "PerlinNoise") {
		palette = CreatePerlinNoisePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::PerlinNoise;
    }
    if (name == "Water") {
		palette = CreateWaterPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::Water;
    }
    if (name == "Sunset") {
		palette = CreateSunsetPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::Sunset;
    }
    if (name == "DeepSpace") {
		palette = CreateDeepSpaceWideVeinsPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::DeepSpace;
    }
    if (name == "Physchodelic") {
		palette = CreatePsychedelicWavePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::Physchodelic;
    }
    if (name == "IceCave") {
		palette = CreateIceCavePalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::IceCave;
    }
    if (name == "AccretionDisk") {
		palette = CreateAccretionDiskPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::AccretionDisk;
    }
    if (name == "ElectricNebula") {
		palette = CreateElectricNebulaPalette(20000);
		paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
		curr_pallete = Palletes::ElectricNebula;
    }
    if (name == "Random") {
        palette = CreateRandomPalette(20000);
        paletteSize = 20000;
        COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
        curr_pallete = Palletes::Random;
    }
}

template <typename Derived>
Palletes FractalBase<Derived>::getPallete() { return curr_pallete; }

template <typename Derived>
sf::Vector2i FractalBase<Derived>::get_resolution() const { return {int(basic_width), int(basic_height)}; }

template <typename Derived>
void FractalBase<Derived>::SetDegreesOffsetForHSV(int degrees) { degrees_offsetForHSV = degrees; setPallete("HSV"); }

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


template <typename Derived>
void FractalBase<Derived>::post_processing() {
    if(!isCudaAvailable){
        image.resize({ basic_width, basic_height }, pixels);
    }
    else if (antialiasing) {
        // SSAA rendering
        dim3 dimBlock(32, 32);
        dim3 dimGrid(
            (width + dimBlock.x - 1) / dimBlock.x,
            (height + dimBlock.y - 1) / dimBlock.y
        );
        if(!custom_formula){
            ANTIALIASING_SSAA4<<<dimGrid, dimBlock, 0, stream>>>(d_pixels, ssaa_buffer, basic_width * 2, basic_height * 2, basic_width, basic_height);
            cudaMemcpyAsync(compressed, ssaa_buffer, basic_width * basic_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        else {
            CU_SAFE_CALL(cuCtxSetCurrent(ctx));
            void* params[] = { &cu_d_pixels, &CUssaa_buffer, &width, &height, &basic_width, &basic_height };
            CU_SAFE_CALL(cuLaunchKernel(kernelAntialiasing,
                    dimGrid.x, dimGrid.y, 1,
                    dimBlock.x, dimBlock.y, 1,
                    0,
                    CUss,
                    params,
                    nullptr
                    ));
            CU_SAFE_CALL(cuMemcpyDtoHAsync((void**)compressed, CUssaa_buffer, basic_height * basic_width * 4 * sizeof(unsigned char),CUss));
            CU_SAFE_CALL(cuStreamSynchronize(CUss));
        }

        image.resize({ basic_width, basic_height }, compressed);

    }
    else {
        if(custom_formula) {
            CU_SAFE_CALL(cuCtxSetCurrent(ctx));
            CU_SAFE_CALL(cuMemcpyDtoHAsync(pixels, cu_d_pixels, basic_width * basic_height * 4 * sizeof(unsigned char), CUss));
            CU_SAFE_CALL(cuMemcpyDtoHAsync(h_total_iterations, cu_d_total_iterations, sizeof(int), CUssData));
            CU_SAFE_CALL(cuStreamSynchronize(CUss));
        }
        else {
            cudaMemcpyAsync(pixels, d_pixels, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_total_iterations, d_total_iterations, sizeof(int), cudaMemcpyDeviceToHost, dataStream);
            // THERE AIN'T NO WAY I AM DOING THAT SYNC LOGIC, JUST LET IT BE
            cudaStreamSynchronize(stream);
        }
        hardness_coeff = *h_total_iterations / (width * height * 1.0);
        image.resize({ basic_width, basic_height }, pixels);
    }
    if(!texture.loadFromImage(image, true)) {
        std::cerr << "Data corrupted!\n";
    }
    sprite.setTexture(texture, true);
    sprite.setPosition({ 0, 0 });
}

// To surely not forget anything lets make sure to delete everything and reallocate
template <typename Derived>
void FractalBase<Derived>::reset() {
    // Free existing resources
    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();

    // Reset parameters
    max_iterations = 300;
    basic_zoom_x = 240.0;
    basic_zoom_y = 240.0;
    zoom_x = basic_zoom_x;
    zoom_y = basic_zoom_y;
    x_offset = 3.0;
    y_offset = 1.85;
    zoom_factor = 1.0;
    zoom_speed = 0.1;
    zoom_scale = 1.0;
    width = 800;
    height = 600;
    basic_width = 800;
    basic_height = 600;

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

/// Formula should be like this one\n
/// new_real = z_real * z_real - z_imag * z_imag + real;\n
/// z_imag =  2 * z_real * z_imag + imag;
/// BEFORE USING THIS FUNCTION MAKE SURE TO SET THE CONTEXT TO NVRTC
template <>
void FractalBase<fractals::mandelbrot>::set_custom_formula(const std::string formula) {
    if(!custom_formula) {
        set_context(context_type::NVRTC);
    }
    CU_SAFE_CALL(cuCtxSetCurrent(ctx));
    kernel_code = R"(
#include "../include/fractals/custom.cuh"
template <typename T>
__global__ void fractal_rendering(
        unsigned char* pixels, unsigned long size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        Color* d_palette, unsigned int paletteSize, T maxIterations, unsigned int* d_total_iterations)
{
    const unsigned int x =   blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y =   blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;
    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }

    const size_t expected_size = width * height * 4;

    const T scale_factor = static_cast<T>(size_of_pixels) / static_cast<T>(expected_size);

    if (x < width && y < height) {
        __shared__ unsigned int total_iterations[1024];
        const T real = x / zoom_x - x_offset;
        const T imag = y / zoom_y - y_offset;
        T new_real = 0.0;
        T z_real = 0.0;
        T z_imag = 0.0;
        T current_iteration = 0;
        T z_comp = z_real * z_real + z_imag * z_imag;

        while (z_comp < 4 && current_iteration < maxIterations) {
)" + formula + R"(
            z_real = new_real;
            z_comp = z_real * z_real + z_imag * z_imag;
            current_iteration++;
        }

        total_iterations[id] = static_cast<unsigned int>(current_iteration);
//        __syncthreads();

        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (id < s) {
                total_iterations[id] += total_iterations[id + s];
            }
//            __syncthreads();
        }
        if (id == 0) {
            //d_total_iterations += total_iterations[0];
            atomicAdd(d_total_iterations, total_iterations[0]);
        }

        unsigned char r, g, b;
        if (current_iteration == maxIterations) {
            r = g = b = 0;
        }

        else {

            //modulus = hypot(z_real, z_imag);
            //double escape_radius = 2.0;
            //if (modulus > escape_radius) {
            //    double nu = log2(log2(modulus) - log2(escape_radius));
            //    current_iteration = current_iteration + 1 - nu;
            //}
            T smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));

            const T cycle_scale_factor = 25.0f;
            T virtual_pos = smooth_iteration * cycle_scale_factor;

            T normalized_pingpong = fmodf(virtual_pos / static_cast<T>(paletteSize -1), 2.0f);
            if (normalized_pingpong < 0.0f) {
                normalized_pingpong += 2.0f;
            }

            T t_interp;
            if (normalized_pingpong <= 1.0f) {
                t_interp = normalized_pingpong;
            } else {
                t_interp = 2.0f - normalized_pingpong;
            }

            T float_index = t_interp * (paletteSize - 1);

            int index1 = static_cast<int>(floorf(float_index));
            int index2 = min(paletteSize - 1, index1 + 1);

            index1 = max(0, index1);

            T t_local = fmodf(float_index, 1.0f);
            if (t_local < 0.0f) t_local += 1.0f;

            Color color1 = getPaletteColor(index1, paletteSize, d_palette);
            Color color2 = getPaletteColor(index2, paletteSize, d_palette);

            float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
            float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
            float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

            r = static_cast<unsigned char>(max(0.0f, min(255.0f, r_f)));
            g = static_cast<unsigned char>(max(0.0f, min(255.0f, g_f)));
            b = static_cast<unsigned char>(max(0.0f, min(255.0f, b_f)));
        }
        const unsigned int base_index = (y * width + x) * 4;
        for (int i = 0; i < scale_factor * 4; i += 4) {
            const unsigned int index = base_index + i;
            pixels[index] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
        }
    }
}

template __global__ void fractal_rendering<float>(
        unsigned char*, size_t, unsigned int, unsigned int,
        float, float, float, float,
        Color*, unsigned int, float, unsigned int*);

template __global__ void fractal_rendering<double>(
        unsigned char*, size_t, unsigned int, unsigned int,
        double, double, double, double,
        Color*, unsigned int, double, unsigned int*);

__global__
void ANTIALIASING_SSAA4(unsigned char* src, unsigned char* dest, unsigned int src_width, unsigned int src_height, unsigned int dest_width, unsigned int dest_height) {
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
)";
    nvrtcProgram prog = nullptr;
    nvrtcResult compileResult;
    std::string lowered_kernel_name_float_str;
    std::string lowered_kernel_name_double_str;
    std::string lowered_kernel_name_ssaa_str;

    try {
        const std::ifstream header_file("../include/fractals/custom.cuh");
        if (!header_file.is_open()) {
            throw std::runtime_error("Could not open header file: custom.cuh");
        }

        std::stringstream buffer;
        buffer << header_file.rdbuf();
        const std::string header_content = buffer.str();

        const char *header_data = header_content.c_str();
        constexpr char *header_name = "custom.cuh";
        const char *headers[] = {header_data};
        const char *includeNames[] = {header_name};

        NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), "custom.cu", 1, headers, includeNames));

        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "fractal_rendering<float>"));
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "fractal_rendering<double>"));
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "ANTIALIASING_SSAA4"));

        std::vector<const char*> compile_options;
        const char* vcpkg_root = std::getenv("VCPKG_ROOT");
        if (vcpkg_root == nullptr) {
            std::cerr << "Warning: VCPKG_ROOT environment variable not set. Assuming a default." << std::endl;
            vcpkg_root = "/home/progamers/vcpkg";
        }
        const std::string triplet = "x64-linux";
        std::string sfml_include_path = "-I" + std::string(vcpkg_root) + "/installed/" + triplet + "/include";
        compile_options.push_back(sfml_include_path.c_str());

        std::string system_include_path1 = "-I/usr/include/c++/14.2.1";
        compile_options.push_back(system_include_path1.c_str());

        std::string system_include_path2 = "-I/usr/include/c++/14.2.1/x86_64-pc-linux-gnu";
        compile_options.push_back(system_include_path2.c_str());

        compile_options.push_back("--gpu-architecture=compute_75");

        const char** opts = compile_options.data();
        int num_opts = compile_options.size();

        compileResult = nvrtcCompileProgram(prog, num_opts, opts);

        size_t logSize = 0;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        }

        std::string log;
        if (logSize > 1) {
            log.resize(logSize);
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
        }

        if (compileResult != NVRTC_SUCCESS) {
            std::cerr << "---------------------\n";
            std::cerr << "NVRTC Compilation Failed:\n";
            std::cerr << "Result Code: " << nvrtcGetErrorString(compileResult) << "\n";
            std::cerr << "---------------------\n";
            std::cerr << "Compilation Log:\n";
            std::cerr << log << std::endl;
            std::cerr << "---------------------\n";

            if (prog) {
                nvrtcDestroyProgram(&prog);
            }
            return;
        } else {
            std::cout << "NVRTC Compilation Succeeded.\n";
            if (!log.empty() && log.length() > 1) {
                std::cout << "---------------------\n";
                std::cout << "Compilation Log (Warnings/Info):\n";
                std::cout << log << std::endl;
                std::cout << "---------------------\n";
            }
        }

        const char* lowered_name_float_ptr = nullptr;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "fractal_rendering<float>", &lowered_name_float_ptr));
        }

        if (lowered_name_float_ptr) {
            lowered_kernel_name_float_str = lowered_name_float_ptr;
        } else {
            // Handle error if name not found after successful compilation (e.g., return)
            std::cerr << "Error: Could not get lowered name for fractal_rendering<float>\n";
            if (prog) nvrtcDestroyProgram(&prog); return;
        }

        const char* lowered_name_double_ptr = nullptr;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "fractal_rendering<double>", &lowered_name_double_ptr));
        }

        if (lowered_name_double_ptr) {
            lowered_kernel_name_double_str = lowered_name_double_ptr;
        } else {
            std::cerr << "Error: Could not get lowered name for fractal_rendering<double>\n";
            if (prog) nvrtcDestroyProgram(&prog); return;
        }

        const char* lowered_name_ssaa_ptr = nullptr;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "ANTIALIASING_SSAA4", &lowered_name_ssaa_ptr));
        }

        if (lowered_name_ssaa_ptr) {
            lowered_kernel_name_ssaa_str = lowered_name_ssaa_ptr;
        } else {
            std::cerr << "Error: Could not get lowered name for ANTIALIASING_SSAA4\n";
            if (prog) nvrtcDestroyProgram(&prog); return;
        }


        size_t ptxSize = 0;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        }

        std::vector<char> ptx;
        if (ptxSize > 0 && prog) {
            ptx.resize(ptxSize);
            NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
        }

        if (prog) {
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
            prog = nullptr;
        }

        if (module_loaded) {
            CU_SAFE_CALL(cuModuleUnload(module));
            module = nullptr;
            module_loaded = false;
        }

        if (!ptx.empty()) {
            CU_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
            module_loaded = true;
        }

        if (module && !lowered_kernel_name_float_str.empty()) {
            CU_SAFE_CALL(cuModuleGetFunction(&kernelFloat, module, lowered_kernel_name_float_str.c_str()));
        } else if (!module) { /* Module not loaded */ } else { /* Name string empty, handle as error */ }


        if (module && !lowered_kernel_name_double_str.empty()) {
            CU_SAFE_CALL(cuModuleGetFunction(&kernelDouble, module, lowered_kernel_name_double_str.c_str()));
        } else if (!module) { /* Module not loaded */ } else { /* Name string empty, handle as error */ }


        if (module && !lowered_kernel_name_ssaa_str.empty()) {
            CU_SAFE_CALL(cuModuleGetFunction(&kernelAntialiasing, module, lowered_kernel_name_ssaa_str.c_str()));
        } else if (!module) { /* Module not loaded */ } else { /* Name string empty, handle as error */ }


    } catch (const std::exception& e) {
        std::cerr << "Exception in NVRTC setup/compilation/loading: " << e.what() << std::endl;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
        }
        throw;
    }
    std::cout << "Kernel loaded successfully.\n";
    custom_formula = true;
}


template <>
/// Formula should be like this one\n
/// new_real = z_real * z_real - z_imag * z_imag + real;\n
/// z_imag =  2 * z_real * z_imag + imag;
/// BEFORE USING THIS FUNCTION MAKE SURE TO SET THE CONTEXT TO NVRTC
void FractalBase<fractals::julia>::set_custom_formula(const std::string formula) {
    if(!custom_formula) {
        set_context(context_type::NVRTC);
    }
    CU_SAFE_CALL(cuCtxSetCurrent(ctx));
    kernel_code = R"(
#include "../include/fractals/custom.cuh"
template <typename T>
__global__ void fractal_rendering_julia(
        unsigned char* pixels, unsigned long size_of_pixels, unsigned int width, unsigned int height,
        T zoom_x, T zoom_y, T x_offset, T y_offset,
        Color* d_palette, unsigned int paletteSize, T maxIterations, unsigned int* d_total_iterations, T cReal, T cImaginary)
{
    const unsigned int x =   blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y =   blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int id = threadIdx.y * blockDim.x + threadIdx.x;
    if (x == 0 && y == 0) {
        *d_total_iterations = 0;
    }

    const size_t expected_size = width * height * 4;

    const T scale_factor = static_cast<T>(size_of_pixels) / static_cast<T>(expected_size);

    if (x < width && y < height) {
        __shared__ unsigned int total_iterations[1024];
        T z_real = x / zoom_x - x_offset;
        T z_imag = y / zoom_y - y_offset;
        T real = cReal;
        T imag = cImaginary;
        T new_real = 0.0;
        T current_iteration = 0;
        T z_comp = z_real * z_real + z_imag * z_imag;

        while (z_comp < 4 && current_iteration < maxIterations) {
)" + formula + R"(
            z_real = new_real;
            z_comp = z_real * z_real + z_imag * z_imag;
            current_iteration++;
        }

        total_iterations[id] = static_cast<unsigned int>(current_iteration);
//        __syncthreads();

        for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (id < s) {
                total_iterations[id] += total_iterations[id + s];
            }
//            __syncthreads();
        }
        if (id == 0) {
            //d_total_iterations += total_iterations[0];
            atomicAdd(d_total_iterations, total_iterations[0]);
        }

        unsigned char r, g, b;
        if (current_iteration == maxIterations) {
            r = g = b = 0;
        }

        else {

            //modulus = hypot(z_real, z_imag);
            //double escape_radius = 2.0;
            //if (modulus > escape_radius) {
            //    double nu = log2(log2(modulus) - log2(escape_radius));
            //    current_iteration = current_iteration + 1 - nu;
            //}
            T smooth_iteration = current_iteration + 1.0f - log2f(log2f(sqrtf(z_real * z_real + z_imag * z_imag)));

            const T cycle_scale_factor = 25.0f;
            T virtual_pos = smooth_iteration * cycle_scale_factor;

            T normalized_pingpong = fmodf(virtual_pos / static_cast<T>(paletteSize -1), 2.0f);
            if (normalized_pingpong < 0.0f) {
                normalized_pingpong += 2.0f;
            }

            T t_interp;
            if (normalized_pingpong <= 1.0f) {
                t_interp = normalized_pingpong;
            } else {
                t_interp = 2.0f - normalized_pingpong;
            }

            T float_index = t_interp * (paletteSize - 1);

            int index1 = static_cast<int>(floorf(float_index));
            int index2 = min(paletteSize - 1, index1 + 1);

            index1 = max(0, index1);

            T t_local = fmodf(float_index, 1.0f);
            if (t_local < 0.0f) t_local += 1.0f;

            Color color1 = getPaletteColor(index1, paletteSize, d_palette);
            Color color2 = getPaletteColor(index2, paletteSize, d_palette);

            float r_f = static_cast<float>(color1.r) + t_local * (static_cast<float>(color2.r) - static_cast<float>(color1.r));
            float g_f = static_cast<float>(color1.g) + t_local * (static_cast<float>(color2.g) - static_cast<float>(color1.g));
            float b_f = static_cast<float>(color1.b) + t_local * (static_cast<float>(color2.b) - static_cast<float>(color1.b));

            r = static_cast<unsigned char>(max(0.0f, min(255.0f, r_f)));
            g = static_cast<unsigned char>(max(0.0f, min(255.0f, g_f)));
            b = static_cast<unsigned char>(max(0.0f, min(255.0f, b_f)));
        }
        const unsigned int base_index = (y * width + x) * 4;
        for (int i = 0; i < scale_factor * 4; i += 4) {
            const unsigned int index = base_index + i;
            pixels[index] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
        }
    }
}

template __global__ void fractal_rendering_julia<float>(
        unsigned char*, size_t, unsigned int, unsigned int,
        float, float, float, float,
        Color*, unsigned int, float, unsigned int*, float, float);

template __global__ void fractal_rendering_julia<double>(
        unsigned char*, size_t, unsigned int, unsigned int,
        double, double, double, double,
        Color*, unsigned int, double, unsigned int*, double, double);

__global__
void ANTIALIASING_SSAA4(unsigned char* src, unsigned char* dest, unsigned int src_width, unsigned int src_height, unsigned int dest_width, unsigned int dest_height) {
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
)";
    nvrtcProgram prog = nullptr;
    nvrtcResult compileResult;
    std::string lowered_kernel_name_float_str;
    std::string lowered_kernel_name_double_str;
    std::string lowered_kernel_name_ssaa_str;

    try {
        const std::ifstream header_file("../include/fractals/custom.cuh");
        if (!header_file.is_open()) {
            throw std::runtime_error("Could not open header file: custom.cuh");
        }

        std::stringstream buffer;
        buffer << header_file.rdbuf();
        const std::string header_content = buffer.str();

        const char *header_data = header_content.c_str();
        constexpr char *header_name = "custom.cuh";
        const char *headers[] = {header_data};
        const char *includeNames[] = {header_name};

        NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), "custom.cu", 1, headers, includeNames));

        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "fractal_rendering_julia<float>"));
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "fractal_rendering_julia<double>"));
        NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, "ANTIALIASING_SSAA4"));

        std::vector<const char*> compile_options;
        const char* vcpkg_root = std::getenv("VCPKG_ROOT");
        if (vcpkg_root == nullptr) {
            std::cerr << "Warning: VCPKG_ROOT environment variable not set. Assuming a default." << std::endl;
            vcpkg_root = "/home/progamers/vcpkg";
        }
        const std::string triplet = "x64-linux";
        std::string sfml_include_path = "-I" + std::string(vcpkg_root) + "/installed/" + triplet + "/include";
        compile_options.push_back(sfml_include_path.c_str());

        std::string system_include_path1 = "-I/usr/include/c++/14.2.1";
        compile_options.push_back(system_include_path1.c_str());

        std::string system_include_path2 = "-I/usr/include/c++/14.2.1/x86_64-pc-linux-gnu";
        compile_options.push_back(system_include_path2.c_str());

        compile_options.push_back("--gpu-architecture=compute_75");

        const char** opts = compile_options.data();
        int num_opts = compile_options.size();

        compileResult = nvrtcCompileProgram(prog, num_opts, opts);

        size_t logSize = 0;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        }

        std::string log;
        if (logSize > 1) {
            log.resize(logSize);
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
        }

        if (compileResult != NVRTC_SUCCESS) {
            std::cerr << "---------------------\n";
            std::cerr << "NVRTC Compilation Failed:\n";
            std::cerr << "Result Code: " << nvrtcGetErrorString(compileResult) << "\n";
            std::cerr << "---------------------\n";
            std::cerr << "Compilation Log:\n";
            std::cerr << log << std::endl;
            std::cerr << "---------------------\n";

            if (prog) {
                NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
            }
            return;
        } else {
            std::cout << "NVRTC Compilation Succeeded.\n";
            if (!log.empty() && log.length() > 1) {
                std::cout << "---------------------\n";
                std::cout << "Compilation Log (Warnings/Info):\n";
                std::cout << log << std::endl;
                std::cout << "---------------------\n";
            }
        }

        const char* lowered_name_float_ptr = nullptr;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "fractal_rendering_julia<float>", &lowered_name_float_ptr));
        }

        if (lowered_name_float_ptr) {
            lowered_kernel_name_float_str = lowered_name_float_ptr;
        } else {
            // Handle error if name not found after successful compilation (e.g., return)
            std::cerr << "Error: Could not get lowered name for fractal_rendering_julia<float>\n";
            if (prog) NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); return;
        }

        const char* lowered_name_double_ptr = nullptr;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "fractal_rendering_julia<double>", &lowered_name_double_ptr));
        }

        if (lowered_name_double_ptr) {
            lowered_kernel_name_double_str = lowered_name_double_ptr;
        } else {
            std::cerr << "Error: Could not get lowered name for fractal_rendering_julia<double>\n";
            if (prog) NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); return;
        }

        const char* lowered_name_ssaa_ptr = nullptr;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, "ANTIALIASING_SSAA4", &lowered_name_ssaa_ptr));
        }

        if (lowered_name_ssaa_ptr) {
            lowered_kernel_name_ssaa_str = lowered_name_ssaa_ptr;
        } else {
            std::cerr << "Error: Could not get lowered name for ANTIALIASING_SSAA4\n";
            if (prog) NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); return;
        }


        size_t ptxSize = 0;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        }

        std::vector<char> ptx;
        if (ptxSize > 0 && prog) {
            ptx.resize(ptxSize);
            NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
        }

        if (prog) {
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
            prog = nullptr;
        }

//        if (module) {
//            CU_SAFE_CALL(cuModuleUnload(module));
//            module = nullptr;
//        }

        if (!ptx.empty()) {
            CU_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
        }

        if (module && !lowered_kernel_name_float_str.empty()) {
            CU_SAFE_CALL(cuModuleGetFunction(&kernelFloat, module, lowered_kernel_name_float_str.c_str()));
        } else if (!module) { /* Module not loaded */ } else { /* Name string empty, handle as error */ }


        if (module && !lowered_kernel_name_double_str.empty()) {
            CU_SAFE_CALL(cuModuleGetFunction(&kernelDouble, module, lowered_kernel_name_double_str.c_str()));
        } else if (!module) { /* Module not loaded */ } else { /* Name string empty, handle as error */ }


        if (module && !lowered_kernel_name_ssaa_str.empty()) {
            CU_SAFE_CALL(cuModuleGetFunction(&kernelAntialiasing, module, lowered_kernel_name_ssaa_str.c_str()));
        } else if (!module) { /* Module not loaded */ } else { /* Name string empty, handle as error */ }


    } catch (const std::exception& e) {
        std::cerr << "Exception in NVRTC setup/compilation/loading: " << e.what() << std::endl;
        if (prog) {
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
        }
        return;
    }
    std::cout << "Kernel loaded successfully.\n";
    custom_formula = true;
}

template <typename Derived>
/// \brief Set the context for the fractal rendering.
/// This function allows switching between CUDA and NVRTC contexts.
/// It frees existing resources and allocates new ones based on the selected context.
void FractalBase<Derived>::set_context(context_type contx) {
    if(contx == context) return;

    // Free existing resources
    FREE_ALL_IMAGE_MEMORY();
    FREE_ALL_NON_IMAGE_MEMORY();

    if(context == context_type::CUDA) {
        if(!initialized_nvrtc){
            CU_SAFE_CALL(cuInit(0));
        }
        initialized_nvrtc = true;
        CU_SAFE_CALL(cuDeviceGet(&device, 0));
        CU_SAFE_CALL(cuCtxCreate(&ctx, 0, device));
        CU_SAFE_CALL(cuCtxSetCurrent(ctx));
        created_context = true;
    }
    else {
        if (ctx) {
            created_context = false;
            CU_SAFE_CALL(cuCtxDestroy(ctx));
            ctx = nullptr;
        }
        if (device) {
            device = 0;
        }
        CUDA_SAFE_CALL(cudaSetDevice(0));
    }
    context = contx;

    ALLOCATE_ALL_IMAGE_MEMORY();
    ALLOCATE_ALL_NON_IMAGE_MEMORY();

    if(context == context_type::NVRTC) custom_formula = true;
    else custom_formula = false;
}

template <typename Derived>
context_type FractalBase<Derived>::get_context() { return context; }



template <typename Derived>
bool FractalBase<Derived>::get_bool_custom_formula() { return custom_formula; }

template <>
void FractalBase<fractals::mandelbrot>::render(render_state quality) {
    if (!isCudaAvailable) {
        // --- CPU Rendering Fallback ---
        // This block executes only if CUDA (GPU acceleration) is not available.

        if (quality == render_state::best) {
            return;
        }

        // --- Global CPU Rendering Lock ---
        // Use an atomic flag to prevent multiple CPU rendering processes from running concurrently.
        bool expected_state = false; // We expect no CPU render to be currently running.
        // Attempt to atomically set g_isCpuRendering to true IF it's currently false.
        // memory_order_acq_rel: Ensures memory operations before this in other threads are visible (acquire)
        //                      and memory operations after this in this thread are visible to others (release).
        //                      Crucial for coordinating access to shared rendering resources.
        if (!g_isCpuRendering.compare_exchange_strong(expected_state, true, std::memory_order_acq_rel)) {
            // If the exchange failed, it means g_isCpuRendering was already true. Another CPU render is active.
            return;
        }

        // --- Launch Background CPU Rendering Thread ---
        // Offload the CPU rendering to a separate thread to avoid blocking the main application thread.
        std::thread main_thread([&]() { // Lambda captures necessary variables by reference.

            // --- Automatic Lock Release ---
            // This RAII guard ensures g_isCpuRendering is set back to false when the lambda finishes
            // execution, even if an exception occurs. This releases the global rendering lock.
            RenderGuard renderGuard(g_isCpuRendering);

            // --- Prepare for New Render Task & Divide Work ---
            unsigned int max_threads_local = std::thread::hardware_concurrency();
            if (max_threads_local == 0) max_threads_local = 1;
            render_targets.clear(); // Clear previous task definitions.

            // Distribute image rows among available threads, handling remainders.
            unsigned int rows_per_thread = basic_height / max_threads_local;
            unsigned int remainder_rows = basic_height % max_threads_local;
            unsigned int current_y_start = 0;
            for (unsigned int i = 0; i < max_threads_local; ++i) {
                unsigned int rows_for_this_thread = rows_per_thread + (i < remainder_rows ? 1 : 0);
                if (rows_for_this_thread == 0 && current_y_start >= basic_height) continue; // Avoid empty tasks
                unsigned int y_start = current_y_start;
                unsigned int y_end = current_y_start + rows_for_this_thread;
                if (y_start < y_end) { // Ensure valid range
                    render_targets.emplace_back(0, y_start, basic_width, y_end);
                    current_y_start = y_end;
                } else if (current_y_start >= basic_height) {
                    break; // Stop if all rows are assigned
                }
            }

            unsigned int actual_threads_to_launch = render_targets.size();

            // --- Thread Limit Check ---
            // Ensure we don't try to launch more threads than we have control flags for.
            if (actual_threads_to_launch > thread_stop_flags.size()) {
                std::cerr << "Warning: Clamping required threads (" << actual_threads_to_launch
                          << ") to available flags (" << thread_stop_flags.size() << ")." << std::endl;
                actual_threads_to_launch = thread_stop_flags.size();
                // This could result in parts of the image not being rendered.
            }

            if (actual_threads_to_launch == 0 && basic_height > 0 && basic_width > 0) {
                // Edge case: image has dimensions, but no threads assigned (e.g., height too small or clamped).
                return; // Exit lambda.
            }

            // --- Launch Worker Threads ---
            for (unsigned int i = 0; i < actual_threads_to_launch; ++i) {
                // Set flag state to '0' (working).
                // memory_order_release: Make this write visible before the thread potentially reads it.
                thread_stop_flags[i].store(0, std::memory_order_release);

                // Launch the actual rendering function (e.g., cpu_render_mandelbrot) in a new thread.
                std::thread t(cpu_render_mandelbrot, render_targets[i], pixels, basic_width, basic_height,
                              zoom_x, zoom_y, x_offset, y_offset, palette.data(), paletteSize,
                              max_iterations, h_total_iterations, std::ref(thread_stop_flags[i])); // Pass flag by ref

                // Detach the worker thread: The main_thread won't wait (join) for it directly.
                // Completion is tracked using the atomic stop flags.
                t.detach();
            }

            // --- Wait for Completion & Intermediate Updates ---
            // This loop polls the status flags of the worker threads.
            while (true) {
                bool all_done = true;
                if (actual_threads_to_launch > 0) {
                    for(unsigned int i = 0; i < actual_threads_to_launch; ++i) {
                        // Check if the flag is '1' (finished).
                        // memory_order_acquire: Ensures reads here see the final writes from the worker.
                        if (thread_stop_flags[i].load(std::memory_order_acquire) != 1) {
                            all_done = false;
                            break;
                        }
                    }
                }

                if (all_done) {
                    break; // All workers finished.
                }

                // Allows updating the display or performing other tasks periodically while rendering.
                post_processing();
                // Sleep briefly to avoid pegging the CPU core running this management thread.
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            // Final update after all threads are confirmed done.
            post_processing();

        }); // End of lambda for main_thread

        // Detach the management thread itself. This makes the entire CPU render operation
        // asynchronous from the perspective of the function caller. The OS handles the detached thread.
        // The RenderGuard inside the lambda ensures the global lock is eventually released.
        main_thread.detach();
        return;

    } // End of if (!isCudaAvailable)
    unsigned int old_width = width, old_height = height;

    double new_zoom_scale;

    if (quality == render_state::good) {
        width = basic_width;
        height = basic_height;
        antialiasing = false;
        new_zoom_scale = 1.0;
    }
    else { // render_state::best, Antialiasing -> more pixels need to be rendered
        width = basic_width * 2;
        height = basic_height * 2;
        antialiasing = true;
        new_zoom_scale = 2.0;
    }

    if (width != old_width || height != old_height) {
        double center_x = x_offset + (old_width / (zoom_x * zoom_scale)) / 2.0;
        double center_y = y_offset + (old_height / (zoom_y * zoom_scale)) / 2.0;

        zoom_scale = new_zoom_scale;

        x_offset = center_x - (width / (zoom_x * zoom_scale)) / 2.0;
        y_offset = center_y - (height / (zoom_y * zoom_scale)) / 2.0;
    }

    double render_zoom_x = zoom_x * zoom_scale;
    double render_zoom_y = zoom_y * zoom_scale;
    size_t len = width * height * 4;
    if(!custom_formula){
        if (render_zoom_x > 1e7) {
            dimBlock = dim3(10, 10);
            dimGrid = dim3(
                    (width + dimBlock.x - 1) / dimBlock.x,
                    (height + dimBlock.y - 1) / dimBlock.y
            );
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations
            );
        }
        else {
            dimBlock = dim3(32, 32);
            dimGrid = dim3(
                    (width + dimBlock.x - 1) / dimBlock.x,
                    (height + dimBlock.y - 1) / dimBlock.y
            );

            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations
            );
        }
    }
    else { // Custom, Formula handling
        cuCtxSetCurrent(ctx);
        dimBlock = dim3(32, 32);
        dimGrid = dim3(
                (width + dimBlock.x - 1) / dimBlock.x,
                (height + dimBlock.y - 1) / dimBlock.y
        );

        size_t len = width * height * 4;
        double render_zoom_x_d = render_zoom_x;
        double render_zoom_y_d = render_zoom_y;
        double x_offset_d = x_offset;
        double y_offset_d = y_offset;
        float max_iterations_val = max_iterations;
        double max_iterations_d = max_iterations;
        unsigned int paletteSize_val = paletteSize;
        unsigned int width_val = width;
        unsigned int height_val = height;

        if (zoom_x > 1e7) {
            void* args[] = {
                    &cu_d_pixels,
                    &len,
                    &width_val,
                    &height_val,
                    &render_zoom_x_d,
                    &render_zoom_y_d,
                    &x_offset_d,
                    &y_offset_d,
                    &cu_palette,
                    &paletteSize_val,
                    &max_iterations_d,
                    &cu_d_total_iterations
            };
            CU_SAFE_CALL(cuLaunchKernel(kernelDouble, dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0, nullptr,
                                        args, nullptr));
        } else { // Запуск float-версии
            float render_zoom_x_f = static_cast<float>(render_zoom_x_d);
            float render_zoom_y_f = static_cast<float>(render_zoom_y_d);
            float x_offset_f = static_cast<float>(x_offset_d);
            float y_offset_f = static_cast<float>(y_offset_d);

            void* args[] = {
                    &cu_d_pixels,
                    &len,
                    &width_val,
                    &height_val,
                    &render_zoom_x_f,
                    &render_zoom_y_f,
                    &x_offset_f,
                    &y_offset_f,
                    &cu_palette,
                    &paletteSize_val,
                    &max_iterations_val,
                    &cu_d_total_iterations
            };
            CU_SAFE_CALL(cuLaunchKernel(kernelFloat,
                                        dimGrid.x, dimGrid.y, 1,
                                        dimBlock.x, dimBlock.y, 1,
                                        0, nullptr,
                                        args, nullptr));
        }

    }


    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess && context != context_type::NVRTC) {
        dimBlock.x -= 2;
        dimBlock.y -= 2;
        dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;
        if (zoom_x > 1e7) {
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                x_offset, y_offset, d_palette, paletteSize,
                max_iterations, d_total_iterations
                );
        }
        else {
            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations
            );
        }
        err = cudaGetLastError();
        if (dimBlock.x < 3) {
            std::cerr << "Critical Issue (mandelbrot set): " << cudaGetErrorString(err) << "\n";
            return;
        }
    }
    ++counter;
    post_processing();
}

template <>
void FractalBase<fractals::julia>::render(
    render_state quality,
    double zx, double zy
) {
    if (!isCudaAvailable) {
        return;
    }

    unsigned int old_width = width, old_height = height;

    double new_zoom_scale;

    if (quality == render_state::good) {
        width = basic_width;
        height = basic_height;
        antialiasing = false;
        new_zoom_scale = 1.0;
    }
    else { // render_state::best, Antialiasing -> more pixels need to be rendered
        width = basic_width * 2;
        height = basic_height * 2;
        antialiasing = true;
        new_zoom_scale = 2.0;
    }

    if (width != old_width || height != old_height) {
        double center_x = x_offset + (old_width / (zoom_x * zoom_scale)) / 2.0;
        double center_y = y_offset + (old_height / (zoom_y * zoom_scale)) / 2.0;

        zoom_scale = new_zoom_scale;

        x_offset = center_x - (width / (zoom_x * zoom_scale)) / 2.0;
        y_offset = center_y - (height / (zoom_y * zoom_scale)) / 2.0;
    }

    double render_zoom_x = zoom_x * zoom_scale;
    double render_zoom_y = zoom_y * zoom_scale;

    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (width + dimBlock.x - 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y
    );


    size_t len = width * height * 4;

    if(!custom_formula){
        if (render_zoom_x > 1e7) {
            dimBlock = dim3(10, 10);
            dimGrid = dim3(
                    (width + dimBlock.x - 1) / dimBlock.x,
                    (height + dimBlock.y - 1) / dimBlock.y
            );
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
        else {
            dimBlock = dim3(32, 32);
            dimGrid = dim3(
                    (width + dimBlock.x - 1) / dimBlock.x,
                    (height + dimBlock.y - 1) / dimBlock.y
            );
            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
    }
    else {
        cuCtxSetCurrent(ctx);
        double maxI = max_iterations;
        float maxI_f = maxI;

        void* args[] = {
                &cu_d_pixels,
                &len,
                &width,
                &height,
                &render_zoom_x,
                &render_zoom_y,
                &x_offset,
                &y_offset,
                &cu_palette,
                &paletteSize,
                &maxI,
                &cu_d_total_iterations,
                &zx,
                &zy
        };

        CUfunction launch_kernel = zoom_x > 1e7 ? kernelDouble : kernelFloat;
        CU_SAFE_CALL(cuLaunchKernel(launch_kernel,
                dimGrid.x, dimGrid.y, 1,
                dimBlock.x, dimBlock.y, 1,
                0, nullptr,
                args, nullptr));
    }

    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess && context != context_type::NVRTC) {
        dimBlock.x -= 2;
        dimBlock.y -= 2;
        dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;
        if (zoom_x > 1e7) {
            fractal_rendering<double><<<dimGrid, dimBlock, 0, stream>>>(
                d_pixels, len, width, height, render_zoom_x, render_zoom_y,
                x_offset, y_offset, d_palette, paletteSize,
                max_iterations, d_total_iterations, zx, zy
            );
        }
        else {
            fractal_rendering<float><<<dimGrid, dimBlock, 0, stream>>>(
                    d_pixels, len, width, height, float(render_zoom_x), float(render_zoom_y),
                    x_offset, y_offset, d_palette, paletteSize,
                    max_iterations, d_total_iterations, zx, zy
            );
        }
        err = cudaGetLastError();
        if (dimBlock.x < 2) {
            std::cout << "Critical Issue (julia set): " << cudaGetErrorString(err) << "\n";
        }
    }
    ++counter;
    post_processing();
}

template <>
void FractalBase<fractals::mandelbrot>::drawIterationLines(sf::Vector2i mouse_pos) {
    int curr_iter = 0;
    double zr = 0.0;
    double zi = 0.0;

    if (max_iterations >= iterationpoints.size()) {
        iterationpoints.resize(max_iterations * 2);
    }

    double cr = mouse_pos.x / zoom_x - x_offset;
    double ci = mouse_pos.y / zoom_y - y_offset;

    while (curr_iter < max_iterations && zr * zr + zi * zi < 4.0) {
        double tmp_zr = zr;
        zr = zr * zr - zi * zi + cr;
        zi = 2.0 * tmp_zr * zi + ci;

        double x = (zr + x_offset) * zoom_x;
        double y = (zi + y_offset) * zoom_y;

        iterationpoints[curr_iter].position = sf::Vector2f(x, y);
        iterationpoints[curr_iter].color = sf::Color::Red;
        curr_iter++;
    }

    iterationline.create(curr_iter);
    iterationline.update(iterationpoints.data());
    drawen_iterations = curr_iter;
}

template <>
void FractalBase<fractals::julia>::drawIterationLines(sf::Vector2i mouse_pos) {
    int curr_iter = 0;
    double zr = 0.0;
    double zi = 0.0;

    // Calculate cr and ci based on the mouse position and current view parameters
    double cr = x_offset + (mouse_pos.x / zoom_x);
    double ci = y_offset + (mouse_pos.y / zoom_y);

    while (curr_iter < max_iterations && zr * zr + zi * zi < 4.0) {
        double tmp_zr = zr;
        zr = zr * zr - zi * zi + cr;
        zi = 2.0 * tmp_zr * zi + ci;

        // Map the complex number z back to screen coordinates
        float x = static_cast<float>((zr - x_offset) * zoom_x);
        float y = static_cast<float>((zi - y_offset) * zoom_y);

        if (curr_iter < iterationpoints.size()) {
            iterationpoints[curr_iter].position = sf::Vector2f(x, y);
            iterationpoints[curr_iter].color = sf::Color::Red;
        }

        curr_iter++;
    }

    iterationline.create(curr_iter);
    iterationline.update(iterationpoints.data());
    drawen_iterations = curr_iter;
}


template <typename Derived>
void FractalBase<Derived>::draw(sf::RenderTarget& target, sf::RenderStates states) const {
     states.transform *= getTransform();
     target.draw(sprite, states);
     target.draw(iterationline, 0, drawen_iterations, states);
}
template <typename Derived>

// Mouse pos should be relative to the picture and not to the screen
void FractalBase<Derived>::handleZoom(double wheel_delta, const sf::Vector2i mouse_pos) {

    double old_zoom_x = zoom_x;
    double old_zoom_y = zoom_y;
    double old_x_offset = x_offset;
    double old_y_offset = y_offset;

    double zoom_change = 1.0 + wheel_delta * zoom_speed;
    zoom_factor *= zoom_change;
    zoom_factor = std::max(std::min(zoom_factor, 100000000000000.0), 0.01);

    zoom_x = basic_zoom_x * zoom_factor;
    zoom_y = basic_zoom_y * zoom_factor;

    double image_mouse_x = mouse_pos.x * 1.0;
    double image_mouse_y = mouse_pos.y * 1.0;


    x_offset = old_x_offset + (image_mouse_x / zoom_x - image_mouse_x / old_zoom_x);
    y_offset = old_y_offset + (image_mouse_y / zoom_y - image_mouse_y / old_zoom_y);

}

template <typename Derived>
void FractalBase<Derived>::start_dragging(sf::Vector2i mouse_pos) {
    is_dragging = true;
    drag_start_pos = mouse_pos;
}


template <typename Derived>
void FractalBase<Derived>::dragging(sf::Vector2i mouse_pos) {
    if (!is_dragging) return;

    sf::Vector2i delta_pos = mouse_pos - drag_start_pos;
    double delta_real = static_cast<double>(delta_pos.x) / (zoom_x * zoom_scale);
    double delta_imag = static_cast<double>(delta_pos.y) / (zoom_y * zoom_scale);

    x_offset += delta_real;
    y_offset += delta_imag;
    drag_start_pos = mouse_pos;
}

template <typename Derived>
void FractalBase<Derived>::stop_dragging() {
    is_dragging = false;
}

template <typename Derived>
void FractalBase<Derived>::move_fractal(sf::Vector2i offset) {
    x_offset += offset.x / (zoom_x * zoom_scale);
	y_offset += offset.y / (zoom_y * zoom_scale);
}

template <>
void FractalBase<fractals::julia>::start_timelapse() {
    timelapse.zx = disX(gen);
    timelapse.zy = disY(gen);
    timelapse.velocityX = disVelX(gen);
	timelapse.velocityY = disVelY(gen);
}

template <>
void FractalBase<fractals::julia>::update_timelapse() {
    const auto elapsed = clock.getElapsedTime();
    const auto elapsedMs = elapsed.asMilliseconds();
    constexpr float frameTimeMs = 1000.0f / 360.0f;

    if (elapsedMs <= frameTimeMs) return;

    float deltaTime = elapsedMs / 1000.0f;
    clock.restart();

    constexpr float GRAVITATIONAL_STRENGTH = 0.1f;
    constexpr float EVENT_HORIZON = 0.1f;
    constexpr float MAX_VEL = 20.0f;
    constexpr float VELOCITY_DAMPING = 0.9995f;
    constexpr float TARGET_CHANGE_TIME = 2.0f;
    constexpr float TRANSITION_SPEED = 3.5f;

    static float timeToTargetChange = TARGET_CHANGE_TIME;
    static float targetZX = 0.0f;
    static float targetZY = 0.0f;
    static float currentTargetZX = 0.0f;
    static float currentTargetZY = 0.0f;

    timeToTargetChange -= deltaTime;
    if (timeToTargetChange <= 0) {
        timeToTargetChange = TARGET_CHANGE_TIME;
        targetZX = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
        targetZY = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
    }

    currentTargetZX += (targetZX - currentTargetZX) * TRANSITION_SPEED * deltaTime;
    currentTargetZY += (targetZY - currentTargetZY) * TRANSITION_SPEED * deltaTime;

    float dx = currentTargetZX - timelapse.zx;
    float dy = currentTargetZY - timelapse.zy;
    float distanceSq = dx * dx + dy * dy + 0.0001f;

    timelapse.velocityX += GRAVITATIONAL_STRENGTH * dx * deltaTime;
    timelapse.velocityY += GRAVITATIONAL_STRENGTH * dy * deltaTime;

    float distance = sqrt(distanceSq);
    if (distance < EVENT_HORIZON) {
        float angle = atan2(dy, dx);
        float boost = (1.0f - (distance / EVENT_HORIZON)) * 4.0f;
        timelapse.velocityX += cos(angle + 3.14159265359f / 2.0f) * boost * deltaTime;
        timelapse.velocityY += sin(angle + 3.14159265359f / 2.0f) * boost * deltaTime;
    }

    timelapse.velocityX *= VELOCITY_DAMPING;
    timelapse.velocityY *= VELOCITY_DAMPING;
    timelapse.velocityX = std::clamp(timelapse.velocityX, -MAX_VEL, MAX_VEL);
    timelapse.velocityY = std::clamp(timelapse.velocityY, -MAX_VEL, MAX_VEL);

    timelapse.zx += timelapse.velocityX * deltaTime;
    timelapse.zy += timelapse.velocityY * deltaTime;

    constexpr float BOUNDARY = 2.0f;
    if (std::abs(timelapse.zx) > BOUNDARY) {
        timelapse.zx = std::copysign(BOUNDARY, timelapse.zx);
        timelapse.velocityX *= -0.5f;
    }
    if (std::abs(timelapse.zy) > BOUNDARY) {
        timelapse.zy = std::copysign(BOUNDARY, timelapse.zy);
        timelapse.velocityY *= -0.5f;
    }

    render(render_state::good, timelapse.zx, timelapse.zy);
}

template <>
void FractalBase<fractals::julia>::stop_timelapse() {
	timelapse.zx = 0;
	timelapse.zy = 0;
	timelapse.velocityX = 0;
	timelapse.velocityY = 0;
}

template class FractalBase<fractals::mandelbrot>;
template class FractalBase<fractals::julia>;