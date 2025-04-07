# CUDA Fractal Renderer

## Description
CUDA Fractal Renderer is a library for rendering fractals (e.g., Mandelbrot) using CUDA for GPU acceleration and SFML for visualization. Optimized for performance, interactivity, and quality, it supports zooming, dragging, and anti-aliasing for seamless fractal exploration.

## Key Features
- 🚀 **CUDA-based GPU rendering** for high speed
- 🔍 **4x SSAA anti-aliasing** for crisp edges
- 🎨 **Dynamic HSV-to-RGB color palette**
- 🖱 **Smooth zoom with mouse wheel**
- 🖱 **Drag-to-explore with mouse**
- ⚡ **Render modes:**
  - **Good** (fast rendering)
  - **Best** (high-quality rendering with AA)
- 🔧 **Extensible for other fractals** (e.g., Julia set)

## Requirements
- **CUDA Toolkit 11.0+** (GPU acceleration)
- **SFML 2.5+** (graphics rendering)
- **C++17 compiler** (GCC, MSVC, Clang, etc.)
- **NVIDIA GPU** (Compute Capability **3.0+**)
- **Windows/Linux** with CUDA support

## Setup
1. **Install CUDA Toolkit:** [Download](https://developer.nvidia.com/cuda-downloads)
2. **Install SFML:** [Download](https://www.sfml-dev.org/download.php)
3. **Clone repository:**
   ```sh
   git clone <repo_url>
   cd cuda-fractal-renderer
   ```
4. **Build with CMake:**
   ```sh
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```
5. **Add 'CUDA files' path to Additional Dependencies in project settings**
6. **Link libraries:** CUDA, SFML (graphics, window, system)

## Usage
```cpp
#include "CUDA files/FractalClass.cuh"
#include <SFML/Graphics.hpp>

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Mandelbrot");
    FractalBase<fractals::mandelbrot> mandelbrot;
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
        }
        mandelbrot.render(render_state::best);  // Render fractal
        window.clear();
        window.draw(mandelbrot); // Draw to screen
        window.display();
    }
    return 0;
}
```

## Controls
- 🔍 **Zoom**: Mouse wheel (in/out)
- 🖱 **Drag**: Left-click + move
- ❌ **Exit**: Close window or press `Esc`

## Performance Notes
- **Good mode**: 800x600, fast render
- **Best mode**: 1600x1200 with AA, higher quality (automatically swithces on this one when have enough time)
- **GPU memory usage** scales with resolution
- **Adjust `max_iterations`** for detail vs. speed

## Extending the Renderer
- **Add new fractal types** via `FractalBase` template
- **Example**: `FractalBase<fractals::julia>`
- **Add new fractals to render** via adding new `<fractal name>.cuh`, `<fractal name>.cu` files

## Support
- **Email**: nobodqwe@gmail.com
- **Issues**: `https://github.com/NeKon69/MandlebrotSetSFML/issues`

