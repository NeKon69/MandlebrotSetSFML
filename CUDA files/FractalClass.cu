#include "FractalClass.cuh"
#include <TGUI/Backend/SFML-Graphics.hpp>
#include <iostream>
#include <thread>
#include <functional>

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
    zoom_scale(1.0), width(400), height(300), sprite(texture)
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
    stopFlagCpu.store(flag);

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
void FractalBase<Derived>::post_processing() {
    if (antialiasing) {
        // SSAA rendering
        dim3 dimBlock(32, 32);
        dim3 dimGrid(
            (width + dimBlock.x - 1) / dimBlock.x,
            (height + dimBlock.y - 1) / dimBlock.y
        );

        auto start = std::chrono::high_resolution_clock::now();
        ANTIALIASING_SSAA4 << <dimGrid, dimBlock, 0, stream >> > (d_pixels, ssaa_buffer, 1600, 1200, 800, 600);
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
        if (status != cudaSuccess && (hardness_coeff > 50) || std::is_same<Derived, fractals::julia>::value) {
            cudaStreamSynchronize(stream);
        }
        image = sf::Image({ 800, 600 }, pixels);
    }
    texture.loadFromImage(image, true);
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
        float(max_iterations), d_total_iterations
        );

    cudaError_t err = cudaGetLastError();
    while (err != cudaSuccess) {
        dimBlock.x -= 2;
        dimBlock.y -= 2;
        dimGrid = (width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y;
        fractal_rendering<<<dimGrid, dimBlock, 0, stream>>>(
            d_pixels, len, width, height, float(render_zoom_x), float(render_zoom_y),
            float(x_offset), float(y_offset), d_palette, paletteSize,
            float(max_iterations), d_total_iterations
            );
        err = cudaGetLastError();
        if (dimBlock.x < 8) {
            std::cout << "Critical Issue (mandelbrot set): " << cudaGetErrorString(err) << "\n";
        }
    }
    ++counter;
    post_processing();
}


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




template <typename Derived>
void FractalBase<Derived>::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    states.transform *= getTransform();
    target.draw(sprite, states);
}
template <typename Derived>
void FractalBase<Derived>::handleZoom(float wheel_delta, const sf::Vector2i mouse_pos) {
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

template class FractalBase<fractals::mandelbrot>;
template class FractalBase<fractals::julia>;