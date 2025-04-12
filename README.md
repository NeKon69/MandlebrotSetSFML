
# CUDA Fractal Explorer

## Description

CUDA Fractal Explorer is a high-performance rendering engine designed for visualizing fractals like the Mandelbrot and Julia sets. It utilizes NVIDIA CUDA for massively parallel GPU computation, SFML for efficient graphics rendering and window management, and TGUI for creating interactive user interface elements.

The project aims to provide a fast, interactive, and visually appealing experience for exploring these complex mathematical structures. It includes a sophisticated demo application featuring a dual-view layout where the Mandelbrot set interactively drives the parameters of the Julia set in real-time.

## Key Features

*   **CUDA-Accelerated Rendering:** Leverages the parallel processing power of NVIDIA GPUs for fast fractal calculations.
*   **Multiple Fractal Algorithms:** Includes implementations for the Mandelbrot set and Julia sets. Designed for straightforward extension to other fractal types.
*   **Interactive Linked View (Demo):** The primary demo (`main.cpp`) displays the Mandelbrot set alongside a Julia set. Moving the mouse over the Mandelbrot set dynamically updates the complex constant (`c`) used to render the Julia set, offering intuitive exploration of the relationship between them.
*   **Adaptive Render Quality:** Implements two rendering modes for balancing performance and visual fidelity:
    *   `render_state::good`: Renders at a base resolution (e.g., 800x600) for fluid interaction during zooming and panning.
    *   `render_state::best`: Renders at a higher internal resolution (e.g., 1600x1200) and applies 4x Super-Sampling Anti-Aliasing (SSAA) for significantly improved image quality when the view is static. The demo automatically transitions between these states.
*   **Extensive Color Palette System:** Features a wide array of built-in coloring algorithms (HSV, Fire, Water, Ice Cave, Accretion Disk, Psychedelic, etc.) selectable via a TGUI ComboBox. Includes a TGUI slider for dynamically adjusting the hue offset in HSV-based palettes.
*   **Standard User Interactions:** Supports intuitive mouse wheel zooming (centered on the cursor position) and left-click-and-drag panning within fractal views.
*   **Iteration Path Visualization:** Holding the right mouse button over the Mandelbrot view displays the escape path for the point under the cursor.
*   **Julia Set Timelapse Mode (Experimental):** A toggleable mode in the demo where the Julia set's parameters (`zx`, `zy`) evolve dynamically over time, simulating movement within the parameter space.
*   **Modern C++ and Build Integration:** Developed using C++20 features, built with CMake, and utilizes vcpkg for streamlined dependency management.

## Technical Stack

*   **Core Languages:** C++20, CUDA C++ (targeting Standard 17)
*   **GPU Acceleration:** NVIDIA CUDA Toolkit
*   **Graphics & Windowing:** SFML 3 (Simple and Fast Multimedia Library)
*   **Graphical User Interface:** TGUI 1 (Texus' Graphical User Interface)
*   **Build System:** CMake (Version 3.20 or higher)
*   **Package Management:** vcpkg

## Requirements

*   **Hardware:**
    *   NVIDIA GPU with Compute Capability 3.0 or higher. (The `CMakeLists.txt` is configured for `sm_75` but can be adapted).
*   **Software:**
    *   NVIDIA CUDA Toolkit (Version compatible with your driver and compiler, e.g., 11.x, 12.x).
    *   C++20 Compliant Compiler (e.g., GCC, Clang, MSVC).
    *   CMake (Version 3.20+).
    *   vcpkg ([vcpkg setup guide](https://vcpkg.io/en/getting-started.html)).
*   **Dependencies (Managed via vcpkg):**
    *   `sfml`: Provides graphics, windowing, and system functionalities.
    *   `tgui`: Provides GUI widgets (ComboBox, Slider).

## Building Instructions

This project uses vcpkg to handle SFML and TGUI dependencies, simplifying the build process.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NeKon69/MandlebrotSetSFML.git
    cd MandlebrotSetSFML
    ```
2.  **Ensure vcpkg is installed and accessible.** The CMake script attempts to find it via `$ENV{HOME}/vcpkg`, but you may need to specify the path explicitly.
3.  **Configure the build using CMake:** Provide the path to your vcpkg toolchain file.
    ```bash
    # Example command, replace <path_to_vcpkg> with your actual vcpkg installation path
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
    ```
4.  **Compile the project:**
    ```bash
    cmake --build build --config Release
    ```
    This will compile the source files and link the necessary libraries specified in `CMakeLists.txt`.

## Running the Demo

After a successful build, the executable (`MandelbrotProject` on Linux/macOS, `MandelbrotProject.exe` on Windows) will be located in the `build` directory (or a subdirectory like `build/Release` depending on the generator). The required font file (`fonts/LiberationMono-Bold.ttf`) is automatically copied to the build directory by CMake.

```bash
cd build
./MandelbrotProject
# On Windows using MSVC, it might be in .\Release\MandelbrotProject.exe
```

## Demo Application Controls

*   **Mouse Wheel:** Zoom in/out. The view under the cursor (Mandelbrot left, Julia right) will be zoomed.
*   **Left Mouse Button + Drag:** Pan the view. Affects the view under the cursor.
*   **Right Mouse Button (Hold, cursor over Mandelbrot):** Visualize the iteration escape path for the selected point.
*   **Mouse Movement (Cursor over Mandelbrot):** Dynamically updates the Julia set parameters (`c = mouse_coords`) if not blocked.
*   **`W`/`A`/`S`/`D` Keys (Cursor over Mandelbrot):** Nudge the Mandelbrot view up/left/down/right.
*   **`B` Key (Cursor over Mandelbrot):** Toggle on/off the updating of the Julia set based on mouse movement over the Mandelbrot set.
*   **`T` Key:** Toggle the experimental Julia Set Timelapse mode.
*   **`Esc` Key:** Exit the application.
*   **TGUI ComboBox (Top Center):** Select the active color palette for both fractal views.
*   **TGUI Slider (Top Center):** Adjust the starting hue offset for HSV-based color palettes.

## Using as a Library

The core fractal rendering logic is encapsulated within the templated `FractalBase` class, allowing it to be integrated into other SFML/TGUI applications.

```cpp
#include "FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <TGUI/TGUI.hpp>
#include <TGUI/Backend/SFML-Graphics.hpp>

int main() {
    sf::RenderWindow window(sf::VideoMode({1024, 768}), "Custom Fractal App");
    tgui::Gui gui(window);

    FractalBase<fractals::mandelbrot> mandelbrotRenderer;
    // mandelbrotRenderer.setPosition({10.f, 10.f}); // Example positioning
    // mandelbrotRenderer.setPallete("ElectricNebula"); // Example palette selection

    sf::Clock deltaClock;
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            gui.handleEvent(*event);
            if (event->is<sf::Event::Closed>())
                window.close();

            // Add custom event handling for zoom, drag, etc.
        }

        // Determine render quality based on interaction state or user preference
        render_state currentQuality = render_state::best; // Or render_state::good

        mandelbrotRenderer.render(currentQuality);

        window.clear(sf::Color::Black);
        window.draw(mandelbrotRenderer); // Draw the fractal
        gui.draw();                     // Draw UI elements
        window.display();
    }

    return 0;
}
```
Ensure your project's build system links against SFML, TGUI (if used), CUDA runtime library (`cudart`), and potentially other libraries like `Threads::Threads`, `X11`, `dl` as shown in the reference `CMakeLists.txt`.

## Performance Notes

*   Render times are heavily dependent on the GPU, the complexity of the current fractal view (especially near the set boundaries), and the `max_iterations` setting.
*   The `Good` rendering state provides significantly faster updates suitable for navigation.
*   The `Best` state involves rendering at 4x the pixel count and performing downsampling (SSAA), requiring substantially more computation but yielding much smoother images.
*   Deep zooms increase the required numerical precision and can significantly impact performance, especially on GPUs with limited double-precision capabilities if double precision were enabled (current implementation mostly uses float, adapting based on zoom).

## Support and Issues

For bug reports, feature requests, or other issues, please use the GitHub issue tracker.

*   **Issue Tracker:** [https://github.com/NeKon69/MandlebrotSetSFML/issues](https://github.com/NeKon69/MandlebrotSetSFML/issues)
*   **Contact:** nobodqwe@gmail.com
