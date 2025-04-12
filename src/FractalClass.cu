#include "FractalClass.cuh"
#include <cuda_runtime.h>
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <iostream>
#include <thread>
#include <random>
#include <math.h>

bool running_other_core = false;

sf::Image stretchImageNearestNeighbor(const sf::Image& source, unsigned int targetWidth, unsigned int targetHeight) {
    sf::Image result({ targetWidth, targetHeight }, sf::Color::Black);

    float scaleX = static_cast<float>(source.getSize().x) / targetWidth;
    float scaleY = static_cast<float>(source.getSize().y) / targetHeight;

    for (unsigned int y = 0; y < targetHeight; ++y) {
        for (unsigned int x = 0; x < targetWidth; ++x) {
            unsigned int srcX = static_cast<unsigned int>(x * scaleX);
            unsigned int srcY = static_cast<unsigned int>(y * scaleY);
            sf::Color color = source.getPixel({ srcX, srcY });
            result.setPixel({ x, y }, color);
        }
    }

    return result;
}

template <typename Derived>
FractalBase<Derived>::FractalBase()
    : max_iterations(300), basic_zoom_x(240.0), basic_zoom_y(240.0),
    zoom_x(basic_zoom_x), zoom_y(basic_zoom_y),
    x_offset(3.0), y_offset(1.85),
    zoom_factor(1.0), zoom_speed(0.1),
    zoom_scale(1.0), width(400), height(300),
    maxComputation(50.f), sprite(texture), iterationline(sf::PrimitiveType::LineStrip),
    gen(rd()), disX(-2.f, 2.f), disY(-1.5f, 1.5f) ,disVelX(-0.13f, 0.13f),
    disVelY(-0.1f, 0.1f)
{
    if (std::is_same<Derived, fractals::julia>::value) {
        x_offset = 2.5;
        //y_offset = 1.25;
        palette = createHSVPalette(20000);
        paletteSize = 20000;
    }
    else {
		palette = createHSVPalette(20000);
        paletteSize = 20000;
    }

    cudaMalloc(&stopFlagDevice, sizeof(bool));
    bool flag = true;
    cudaMemcpy(stopFlagDevice, &flag, sizeof(bool), cudaMemcpyHostToDevice);

    cudaMalloc(&d_palette, palette.size() * sizeof(sf::Color));
    cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);

    // Alloc data for the GPU uncompressed image
    cudaMallocManaged(&d_pixels, 1600 * 1200 * 4 * sizeof(uint32_t));

    // Alloc data for the CPU uncompressed image
    cudaMallocHost(&pixels, 1600 * 1200 * 4 * sizeof(char4));

    // Alloc data for the GPU compressed image
    cudaMalloc(&ssaa_buffer, 800 * 600 * 4 * sizeof(char4));

    // Alloc data for the CPU compressed image
	cudaMallocHost(&compressed, 800 * 600 * 4 * sizeof(char4));

	cudaStreamCreate(&stream);

    cudaEventCreate(&start_rendering);
    cudaEventCreate(&stop_rendering);

    cudaMalloc(&d_total_iterations, sizeof(int));
    cudaMemset(d_total_iterations, 0, sizeof(int));
    cudaMallocHost(&h_total_iterations , sizeof(int));

    cudaStreamCreate(&dataStream);

    iterationpoints.resize(max_iterations);
}

template <typename Derived>
FractalBase<Derived>::~FractalBase() {
    cudaFree(d_pixels);
    cudaFree(d_palette);
    cudaFree(stopFlagDevice);
    cudaFree(d_total_iterations);
    cudaFreeHost(pixels);
	cudaFreeHost(compressed);
    cudaFreeHost(h_total_iterations);
	cudaEventDestroy(start_rendering);
	cudaEventDestroy(stop_rendering);
    cudaStreamDestroy(stream);
	cudaStreamDestroy(dataStream);
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
void FractalBase<Derived>::setMaxComputation(float Gflops) { maxComputation = 50.0f / 90 * Gflops; }

template <typename Derived>
void FractalBase<Derived>::setPallete(std::string name) { 
    if (name == "HSV") {
		palette = createHSVPalette(20000, degrees_offsetForHSV);
		paletteSize = 20000;
        cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
        curr_pallete = Palletes::HSV;
    }
    if (name == "Basic") {
		palette = BluePlusBlackWhitePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::Basic;
    }
    if (name == "BlackOWhite") {
		palette = CreateBlackOWhitePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::BlackOWhite;
    }
    if (name == "OscillatingGrayscale") {
		palette = CreateOscillatingGrayscalePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::OscillatingGrayscale;
    }
    if (name == "Interpolated") {
		palette = CreateInterpolatedPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::Interpolated;
    }
    if (name == "Pastel") {
        palette = CreatePastelPalette(20000);
        paletteSize = 20000;
        cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
        curr_pallete = Palletes::Pastel;
    }
    if (name == "CyclicHSV") {
		palette = CreateCyclicHSVPpalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::CyclicHSV;
    }
    if (name == "Fire") {
		palette = CreateFirePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::Fire;
    }
    if (name == "FractalPattern") {
		palette = CreateFractalPatternPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::FractalPattern;
    }
    if (name == "PerlinNoise") {
		palette = CreatePerlinNoisePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::PerlinNoise;
    }
    if (name == "Water") {
		palette = CreateWaterPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::Water;
    }
    if (name == "Sunset") {
		palette = CreateSunsetPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::Sunset;
    }
    if (name == "DeepSpace") {
		palette = CreateDeepSpaceWideVeinsPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::DeepSpace;
    }
    if (name == "Physchodelic") {
		palette = CreatePsychedelicWavePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::Physchodelic;
    }
    if (name == "IceCave") {
		palette = CreateIceCavePalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::IceCave;
    }
    if (name == "AccretionDisk") {
		palette = CreateAccretionDiskPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::AccretionDisk;
    }
    if (name == "ElectricNebula") {
		palette = CreateElectricNebulaPalette(20000);
		paletteSize = 20000;
		cudaMemcpy(d_palette, palette.data(), palette.size() * sizeof(sf::Color), cudaMemcpyHostToDevice);
		curr_pallete = Palletes::ElectricNebula;
    }
}

template <typename Derived>
Palletes FractalBase<Derived>::getPallete() { return curr_pallete; }

template <typename Derived>
void FractalBase<Derived>::SetDegreesOffsetForHSV(int degrees) { degrees_offsetForHSV = degrees; setPallete("HSV"); }

template <typename Derived>
void FractalBase<Derived>::post_processing() {
    if (antialiasing) {
        // SSAA rendering
        dim3 dimBlock(32, 32);
        dim3 dimGrid(
            (width + dimBlock.x - 1) / dimBlock.x,
            (height + dimBlock.y - 1) / dimBlock.y
        );

        auto start = std::chrono::high_resolution_clock::now();
        ANTIALIASING_SSAA4<<<dimGrid, dimBlock, 0, stream>>>(d_pixels, ssaa_buffer, 1600, 1200, 800, 600);
        auto end = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(compressed, ssaa_buffer, 800 * 600 * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);

        std::cout << "SSAA4 time + copying to host: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        cudaStreamSynchronize(stream);
        image = sf::Image({ 800, 600 }, compressed);
    }
    else {
        cudaEventRecord(stop_rendering, stream);
        cudaMemcpyAsync(pixels, d_pixels, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_total_iterations, d_total_iterations, sizeof(int), cudaMemcpyDeviceToHost, dataStream);
        cudaError_t status = cudaEventQuery(stop_rendering);
        hardness_coeff = *h_total_iterations / (width * height * 1.0);
        if (status != cudaSuccess && (hardness_coeff > maxComputation) || std::is_same<Derived, fractals::julia>::value || zoom_x > 1e7) {
            cudaStreamSynchronize(stream);
        }
        image = sf::Image({ 800, 600 }, pixels);
    }
    texture = sf::Texture(image, true);
    sprite.setTexture(texture, true);
    sprite.setPosition({ 0, 0 });
}

// that code served me good in the past, however it's being replaced with better version with atomic operations
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
template <>
void FractalBase<fractals::mandelbrot>::render(render_state quality) {
    cudaEvent_t event;
    cudaEventCreate(&event);

    int new_width, new_height;
    double new_zoom_scale;

    if (quality == render_state::good) {
        new_width = 800;
        new_height = 600;
        antialiasing = false;
        new_zoom_scale = 1.0;
    }
    else { // render_state::best
        new_width = 1600;
        new_height = 1200;
        antialiasing = true;
        new_zoom_scale = 2.0;
    }

    if (width != new_width || height != new_height) {
        double center_x = x_offset + (width / (zoom_x * zoom_scale)) / 2.0;
        double center_y = y_offset + (height / (zoom_y * zoom_scale)) / 2.0;

        zoom_scale = new_zoom_scale;
        width = new_width;
        height = new_height;

        x_offset = center_x - (width / (zoom_x * zoom_scale)) / 2.0;
        y_offset = center_y - (height / (zoom_y * zoom_scale)) / 2.0;

        width = new_width;
        height = new_height;
    }

    double render_zoom_x = zoom_x * zoom_scale;
    double render_zoom_y = zoom_y * zoom_scale;
    size_t len = width * height * 4;
    cudaEventRecord(start_rendering, stream);
    if (render_zoom_x > 1e7) {
        dimBlock = dim3(10, 10);
        dimGrid = dim3(
            (width + dimBlock.x - 1) / dimBlock.x,
            (height + dimBlock.y - 1) / dimBlock.y
        );
        fractal_rendering<<<dimGrid, dimBlock, 0, stream>>>(
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

        fractal_rendering<<<dimGrid, dimBlock, 0, stream>>>(
            d_pixels, len, width, height, float(render_zoom_x), float(render_zoom_y),
            float(x_offset), float(y_offset), d_palette, paletteSize,
            float(max_iterations), d_total_iterations
            );
    }

    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess) {
        dimBlock.x -= 2;
        dimBlock.y -= 2;
        dimGrid = (width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y;
        fractal_rendering<<<dimGrid, dimBlock, 0, stream>>>(
            d_pixels, len, width, height, render_zoom_x, render_zoom_y,
            x_offset, y_offset, d_palette, paletteSize,
            max_iterations, d_total_iterations
            );
        err = cudaGetLastError();
        if (dimBlock.x < 8) {
            std::cout << "Critical Issue (mandelbrot set): " << cudaGetErrorString(err) << "\n";
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
    cudaEvent_t event;
    cudaEventCreate(&event);

    int new_width, new_height;
    double new_zoom_scale;

    if (quality == render_state::good) {
        new_width = 800;
        new_height = 600;
        antialiasing = false;
        new_zoom_scale = 1.0;
    }
    else { // render_state::best
        new_width = 1600;
        new_height = 1200;
        antialiasing = true;
        new_zoom_scale = 2.0;
    }

    if (width != new_width || height != new_height) {
        double center_x = x_offset + (width / (zoom_x * zoom_scale)) / 2.0;
        double center_y = y_offset + (height / (zoom_y * zoom_scale)) / 2.0;

        zoom_scale = new_zoom_scale;
        width = new_width;
        height = new_height;

        x_offset = center_x - (width / (zoom_x * zoom_scale)) / 2.0;
        y_offset = center_y - (height / (zoom_y * zoom_scale)) / 2.0;

        width = new_width;
        height = new_height;
    }

    double render_zoom_x = zoom_x * zoom_scale;
    double render_zoom_y = zoom_y * zoom_scale;

    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (width + dimBlock.x - 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y
    );


    size_t len = width * height * 4;
    cudaEventRecord(start_rendering, stream);
    fractal_rendering<<<dimGrid, dimBlock, 0, stream>>>(
        d_pixels, len, width, height, float(render_zoom_x), float(render_zoom_y),
        float(x_offset), float(y_offset), d_palette, paletteSize,
        float(max_iterations), d_total_iterations, zx, zy
        );

    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess) {
        dimBlock.x -= 2;
        dimBlock.y -= 2;
        dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;
        fractal_rendering<<<dimGrid, dimBlock, 0, stream>>>(
            d_pixels, len, width, height, float(render_zoom_x), float(render_zoom_y),
            float(x_offset), float(y_offset), d_palette, paletteSize,
            float(max_iterations), d_total_iterations, zx, zy
            );
        err = cudaGetLastError();
        if (dimBlock.x < 8) {
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
    if (std::is_same<Derived, fractals::julia>::value) {
        image_mouse_x -= 1920 - 800;
    }

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

float getBoundedVelocity(float currentPos, float currentVel, float minPos, float maxPos, float minVel, float maxVel) {
    const float margin = 0.2f;

    if (currentPos > maxPos - margin && currentVel > 0) {
        return std::max(-maxVel, currentVel * -0.5f);
    }

    else if (currentPos < minPos + margin && currentVel < 0) {
        return std::min(maxVel, currentVel * -0.5f);
    }

    return minVel + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxVel - minVel)));
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