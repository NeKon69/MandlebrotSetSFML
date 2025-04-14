#include <TGUI/TGUI.hpp>
#include "FractalClass.cuh" // Assuming FractalBase, fractals::*, render_state, Palletes defined here
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <ctime> // For time formatting

// Alias for clarity
using RenderQuality = render_state;

void main_thread() {
    // --- Interaction State Variables ---
    float mandelbrotIdleTimeMs = 0.0f;
    sf::Time juliaIdleTime = sf::Time::Zero;
    sf::Clock frameClock;

    // --- Render State Variables ---
    RenderQuality currentMandelbrotQuality = RenderQuality::good;
    RenderQuality previousMandelbrotQuality = RenderQuality::good; // Tracks the quality used in the *last* render
    RenderQuality currentJuliaQuality = RenderQuality::good;
    RenderQuality previousJuliaQuality = RenderQuality::good;    // Tracks the quality used in the *last* render

    // --- Render Request Flags ---
    bool needsMandelbrotRender = true;
    bool needsJuliaRender = true;
    bool showMandelbrotIterationLines = false;

    // --- SFML Window and Render Target Setup ---
    const sf::Vector2u windowSize = { 1920, 1080 };
    sf::RenderWindow window(sf::VideoMode(windowSize), "Fractals");
    sf::RenderTexture renderTarget;
     renderTarget = sf::RenderTexture({windowSize.x, windowSize.y});


    // --- Font and Text Elements for UI Overlays ---
    sf::Font mainFont;
    if (!mainFont.openFromFile("fonts/LiberationMono-Bold.ttf")) {
         std::cerr << "Error loading font!" << std::endl;
        return;
    }
    sf::Text fpsDisplay(mainFont);
    fpsDisplay.setPosition({ 10, 10 });
    fpsDisplay.setString("FPS: 0");
    sf::Text mandelbrotHardnessDisplay(mainFont);
    mandelbrotHardnessDisplay.setPosition({ 10, 40 });
    mandelbrotHardnessDisplay.setString("M-Hardness: 0.0");
    sf::Text juliaHardnessDisplay(mainFont);
    juliaHardnessDisplay.setPosition({ static_cast<float>(windowSize.x - 800 + 10), 40.f }); // Position relative to Julia area
    juliaHardnessDisplay.setString("J-Hardness: 0.0");

    // --- Mouse State ---
    sf::Vector2i mousePosition;

    // --- Julia Set Parameters ---
    double juliaSeedReal = 0.0;
    double juliaSeedImag = 0.0;

    // --- Core Application State Flags ---
    bool isJuliaVisible = false;
    bool isLeftMouseDownMandelbrot = false; // Tracks if left mouse is held in Mandelbrot area
    bool isLeftMouseDownJulia = false;      // Tracks if left mouse is held in Julia area
    bool isTimelapseActive = false;
    bool blockJuliaParameterUpdate = false;

    // --- Per-Frame Event Flags (Reset each loop iteration) ---
    bool isZoomingMandelbrot = false;
    bool mouseMovedInMandelbrotArea = true; // Start true for initial render
    bool isZoomingJulia = false;
    bool mouseMovedInJuliaArea = false;

    // --- Timers ---
    sf::Clock mandelbrotRenderClock;
    sf::Clock juliaRenderClock;
    sf::Clock fpsUpdateClock;
    sf::Clock qualityImprovementClock; // Tracks time since last interaction for quality upgrades

    unsigned int frameCountForFPS = 0;

    // --- Fractal Objects Instantiation ---
    using MandelbrotFractal = FractalBase<fractals::mandelbrot>;
    using JuliaFractal = FractalBase<fractals::julia>;

    MandelbrotFractal mandelbrotFractal;
    JuliaFractal juliaFractal;
    juliaFractal.setPosition({ static_cast<float>(windowSize.x - 800), 0.f }); // Position Julia set sprite

    window.setFramerateLimit(360);

    // --- Performance Measurement ---
    std::cout << "Measuring GFLOPS..." << std::endl;
    float measuredGflops = measureGFLOPS(50000); // Assuming measureGFLOPS is defined elsewhere
    std::cout << "Estimated GFLOPS: " << measuredGflops << std::endl;
    mandelbrotFractal.setMaxComputation(measuredGflops);
    juliaFractal.setMaxComputation(measuredGflops);

    // --- TGUI Setup ---
    tgui::Gui gui(window);

    tgui::Slider::Ptr paletteOffsetSlider = tgui::Slider::create();
    paletteOffsetSlider->setMinimum(0);
    paletteOffsetSlider->setMaximum(360);
    paletteOffsetSlider->onValueChange([&](float pos) {
        // Apply offset only if an HSV-based palette is selected
        if (mandelbrotFractal.getPallete() == Palletes::HSV || mandelbrotFractal.getPallete() == Palletes::CyclicHSV)
            mandelbrotFractal.SetDegreesOffsetForHSV(static_cast<int>(pos));
        if (juliaFractal.getPallete() == Palletes::HSV || juliaFractal.getPallete() == Palletes::CyclicHSV)
            juliaFractal.SetDegreesOffsetForHSV(static_cast<int>(pos));
        needsMandelbrotRender = true;
        needsJuliaRender = true;
        });
    paletteOffsetSlider->setPosition({ 810, 40 });
    paletteOffsetSlider->setSize({200, 18});

    tgui::ComboBox::Ptr paletteSelector = tgui::ComboBox::create();
    paletteSelector->setPosition({ 810, 10 });
    paletteSelector->setSize({200, 24});
    const std::vector<std::string> paletteNames = {
        "HSV", "BlackOWhite", "Basic", "OscillatingGrayscale", "Fire", "Pastel",
        "Interpolated", "CyclicHSV", "FractalPattern", "PerlinNoise", "Water",
        "Sunset", "DeepSpace", "Physchodelic", "IceCave", "ElectricNebula",
        "AccretionDisk"};
    for(const auto& name : paletteNames) {
        paletteSelector->addItem(name);
    }
    paletteSelector->setSelectedItem("HSV");
    paletteSelector->onItemSelect([&](const tgui::String& str) {
        needsMandelbrotRender = true;
        mandelbrotFractal.setPallete(str.toStdString());
        needsJuliaRender = true;
        juliaFractal.setPallete(str.toStdString());
        });

    gui.add(paletteSelector);
    gui.add(paletteOffsetSlider);

    // ========================================================================
    // --- Main Render Loop ---
    // ========================================================================
    while (window.isOpen()) {

        // --------------------------------------------------------------------
        // --- Event Handling ---
        // --------------------------------------------------------------------
        while (const auto event = window.pollEvent()) {
             gui.handleEvent(*event);

             // Reset quality improvement timer on any interaction
             if (event->is<sf::Event::MouseWheelScrolled>() ||
                event->is<sf::Event::MouseMoved>() ||
                event->is<sf::Event::KeyPressed>() ||
                event->is<sf::Event::MouseButtonPressed>())
            {
                 qualityImprovementClock.restart();
            }

            // --- Mouse Movement ---
            if (const auto* mouseMoveEvent = event->getIf<sf::Event::MouseMoved>()) {
                mousePosition = mouseMoveEvent->position;

                // Define approximate areas for interaction
                bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < windowSize.y; // Adjust 800 based on actual Mandelbrot size/layout
                bool isInJuliaArea = mousePosition.x >= windowSize.x - 800 && mousePosition.x < windowSize.x && mousePosition.y >= 0 && mousePosition.y < windowSize.y; // Adjust 800 based on actual Julia size/layout


                if (isInMandelbrotArea) {
                    mouseMovedInMandelbrotArea = true;
                    if (showMandelbrotIterationLines) {
                        mandelbrotFractal.drawIterationLines(mousePosition);
                        needsMandelbrotRender = true;
                    }
                    if (mandelbrotFractal.get_is_dragging()) { // Check internal state
                        currentMandelbrotQuality = RenderQuality::good;
                        mandelbrotFractal.dragging(mousePosition);
                        needsMandelbrotRender = true;
                    }
                }
                else if (isInJuliaArea) {
                    mouseMovedInJuliaArea = true;
                     if (juliaFractal.get_is_dragging()) { // Check internal state
                        currentJuliaQuality = RenderQuality::good;
                        juliaFractal.dragging(mousePosition);
                        needsJuliaRender = true;
                     }
                }
            }

            // --- Window Close ---
            if (event->is<sf::Event::Closed>()) {
                window.close();
                break;
            }

            // --- Keyboard Input ---
            if (const auto* keyPressEvent = event->getIf<sf::Event::KeyPressed>()) {
                 // Check location for context-sensitive keys (using fixed coordinates)
                 bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < windowSize.y;

                switch(keyPressEvent->scancode) {
                    case sf::Keyboard::Scancode::Escape:
                        window.close();
                        break;

                    case sf::Keyboard::Scancode::R: // Increase iterations
                        if (isInMandelbrotArea) {
                            mandelbrotFractal.set_max_iters(mandelbrotFractal.get_max_iters() * 2);
                            needsMandelbrotRender = true;
                            currentMandelbrotQuality = RenderQuality::best;
                        }
                        break;

                    case sf::Keyboard::Scancode::B: // Toggle Julia parameter lock
                        if (isInMandelbrotArea) {
                            blockJuliaParameterUpdate = !blockJuliaParameterUpdate;
                             if (!blockJuliaParameterUpdate) needsJuliaRender = true; // Render if unblocked
                        }
                        break;

                    // Fractal Movement Keys
                    case sf::Keyboard::Scancode::W:
                    case sf::Keyboard::Scancode::S:
                    case sf::Keyboard::Scancode::A:
                    case sf::Keyboard::Scancode::D:
                        if (isInMandelbrotArea) {
                            sf::Vector2f direction = {0.f, 0.f};
                            if (keyPressEvent->scancode == sf::Keyboard::Scancode::W) direction.y = 1.f;
                            if (keyPressEvent->scancode == sf::Keyboard::Scancode::S) direction.y = -1.f;
                            if (keyPressEvent->scancode == sf::Keyboard::Scancode::A) direction.x = 1.f;
                            if (keyPressEvent->scancode == sf::Keyboard::Scancode::D) direction.x = -1.f;

                            mouseMovedInMandelbrotArea = true;
                            currentMandelbrotQuality = RenderQuality::good;
                            mandelbrotFractal.move_fractal({int(direction.x), int(direction.y)});
                        }
                        break;

                    case sf::Keyboard::Scancode::T: // Toggle Timelapse
                        isTimelapseActive = !isTimelapseActive;
                        if(isTimelapseActive) {
                             juliaFractal.start_timelapse();
                             std::cout << "Timelapse started." << std::endl;
                        } else {
                             juliaFractal.stop_timelapse();
                             std::cout << "Timelapse stopped." << std::endl;
                             needsJuliaRender = true;
                             currentJuliaQuality = RenderQuality::best;
                        }
                        break;

                    case sf::Keyboard::Scancode::F6: // Screenshot Mandelbrot
                        {
                            sf::Texture texture;
                            // Ensure texture size matches the fractal's rendered texture
                            texture = sf::Texture({mandelbrotFractal.getTexture().getSize().x, mandelbrotFractal.getTexture().getSize().y});
                            texture.update(mandelbrotFractal.getTexture());
                            sf::Image screenshot = texture.copyToImage();

                            auto now = std::chrono::system_clock::now();
                            std::time_t timer = std::chrono::system_clock::to_time_t(now);
                            std::tm time_info;
                            char filenameBuffer[80];

                            #ifdef _WIN32
                            localtime_s(&time_info, &timer);
                            #else
                            localtime_r(&timer, &time_info); // POSIX
                            #endif
                            std::strftime(filenameBuffer, sizeof(filenameBuffer), "%Y-%m-%d_%H-%M-%S", &time_info);

                            std::string filename = "mandelbrot_";
                            filename += filenameBuffer;
                            filename += ".png";

                            if (screenshot.saveToFile(filename)) {
                                std::cout << "Screenshot saved to " << filename << std::endl;
                            } else {
                                std::cerr << "Error: Failed to save screenshot to " << filename << std::endl;
                            }
                        }
                        break;
                    default:
                        break;
                }
            }

            // --- Mouse Wheel Scroll (Zooming) ---
            if (const auto* mouseWheelEvent = event->getIf<sf::Event::MouseWheelScrolled>()) {
                float delta = mouseWheelEvent->delta;
                // Define approximate areas for interaction
                bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < windowSize.y;
                bool isInJuliaArea = mousePosition.x >= windowSize.x - 800 && mousePosition.x < windowSize.x && mousePosition.y >= 0 && mousePosition.y < windowSize.y;

                if (isInMandelbrotArea) {
                    isZoomingMandelbrot = true;
                    currentMandelbrotQuality = RenderQuality::good;
                    mandelbrotFractal.handleZoom(delta, mousePosition);
                    needsMandelbrotRender = true;
                }
                else if (isInJuliaArea) {
                    isZoomingJulia = true;
                    currentJuliaQuality = RenderQuality::good;
                    juliaFractal.handleZoom(delta, mousePosition);
                    needsJuliaRender = true;
                }
            }

            // --- Mouse Button Pressed ---
            if (const auto* mouseButtonPressedEvent = event->getIf<sf::Event::MouseButtonPressed>()) {
                 // Define approximate areas for interaction
                 bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < windowSize.y;
                 bool isInJuliaArea = mousePosition.x >= windowSize.x - 800 && mousePosition.x < windowSize.x && mousePosition.y >= 0 && mousePosition.y < windowSize.y;

                if (mouseButtonPressedEvent->button == sf::Mouse::Button::Left) {
                    if (isInMandelbrotArea) {
                        isLeftMouseDownMandelbrot = true;
                        currentMandelbrotQuality = RenderQuality::good;
                        mandelbrotFractal.start_dragging(mousePosition);
                    } else if (isInJuliaArea) {
                        isLeftMouseDownJulia = true;
                        currentJuliaQuality = RenderQuality::good;
                        juliaFractal.start_dragging(mousePosition);
                    }
                } else if (mouseButtonPressedEvent->button == sf::Mouse::Button::Right) { // Right Click for lines
                    if (isInMandelbrotArea) {
                        showMandelbrotIterationLines = true;
                        mandelbrotFractal.drawIterationLines(mousePosition);
                        needsMandelbrotRender = true;
                    }
                }
            }

            // --- Mouse Button Released ---
            if (const auto* mouseButtonReleasedEvent = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleasedEvent->button == sf::Mouse::Button::Left) {
                    if (isLeftMouseDownMandelbrot) {
                        isLeftMouseDownMandelbrot = false;
                        currentMandelbrotQuality = RenderQuality::best;
                        mandelbrotFractal.stop_dragging();
                        needsMandelbrotRender = true; // Explicit render request after drag
                    }
                    if (isLeftMouseDownJulia) {
                        isLeftMouseDownJulia = false;
                        currentJuliaQuality = RenderQuality::best;
                        juliaFractal.stop_dragging();
                        needsJuliaRender = true; // Explicit render request after drag
                    }
                } else if (mouseButtonReleasedEvent->button == sf::Mouse::Button::Right) { // Right Click for lines
                     bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < windowSize.y;
                     if (isInMandelbrotArea) {
                         showMandelbrotIterationLines = false;
                         needsMandelbrotRender = true; // Rerender to remove lines
                     }
                }
            }

        } // --- End Event Handling Loop ---


        // ========================================================================
        // --- Update and Render Logic ---
        // ========================================================================

        sf::Time deltaTime = frameClock.restart();
        frameCountForFPS++;

        // --------------------------------------------------------------------
        // --- Mandelbrot Update and Render Decision ---
        // --------------------------------------------------------------------

        bool isDraggingMandelbrotThisFrame = isLeftMouseDownMandelbrot && mouseMovedInMandelbrotArea;
        bool isZoomingMandelbrotThisFrame = isZoomingMandelbrot;
        bool isInteractingMandelbrotThisFrame = isDraggingMandelbrotThisFrame || isZoomingMandelbrotThisFrame;

        if (isInteractingMandelbrotThisFrame) {
            mandelbrotIdleTimeMs = 0.0f;
        } else {
            mandelbrotIdleTimeMs += deltaTime.asMilliseconds();
        }

        bool renderMandelbrotThisFrame = false;
        RenderQuality qualityForMandelbrotRender = currentMandelbrotQuality;

        // Determine if a render is needed based on flags, interaction, or idle time
        if (needsMandelbrotRender) {
            renderMandelbrotThisFrame = true;
            qualityForMandelbrotRender = (currentMandelbrotQuality == RenderQuality::best) ? RenderQuality::best : RenderQuality::good;
        }
        if (isInteractingMandelbrotThisFrame) {
            renderMandelbrotThisFrame = true;
            qualityForMandelbrotRender = RenderQuality::good; // Force lower quality during interaction
        }
        else if (!renderMandelbrotThisFrame) { // Check for idle improvement only if not already rendering
            const float IDLE_TIME_THRESHOLD_MS = 500.0f;
            bool needsQualityImprovement = mandelbrotIdleTimeMs > IDLE_TIME_THRESHOLD_MS;
            if (needsQualityImprovement && (currentMandelbrotQuality != RenderQuality::best || previousMandelbrotQuality != RenderQuality::best)) {
                renderMandelbrotThisFrame = true;
                qualityForMandelbrotRender = RenderQuality::best; // Improve quality after idle period
            }
        }

        // Perform Mandelbrot Render
        if (renderMandelbrotThisFrame)
        {
            needsMandelbrotRender = false; // Consume the flag
            currentMandelbrotQuality = qualityForMandelbrotRender;
            previousMandelbrotQuality = currentMandelbrotQuality;

            // mandelbrotRenderClock.restart();
            mandelbrotFractal.render(currentMandelbrotQuality);
            // auto renderDuration = mandelbrotRenderClock.getElapsedTime();
            // std::cout << "Mandelbrot render time: " << renderDuration.asMilliseconds() << " ms" << std::endl;
        }


        // --------------------------------------------------------------------
        // --- Julia Update and Render Decision ---
        // --------------------------------------------------------------------

        bool isDraggingJuliaThisFrame = isLeftMouseDownJulia && mouseMovedInJuliaArea;
        bool isZoomingJuliaThisFrame = isZoomingJulia;
        bool isInteractingJuliaThisFrame = isDraggingJuliaThisFrame || isZoomingJuliaThisFrame;

        if (isInteractingJuliaThisFrame) {
            juliaIdleTime = sf::Time::Zero;
        } else {
            juliaIdleTime += deltaTime;
        }

        bool renderJuliaThisFrame = false;
        RenderQuality qualityForJuliaRender = currentJuliaQuality;
        // Define approximate area for mouse interaction triggering Julia update
        bool isMouseInMandelbrotUpdateArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < windowSize.y;


        // Determine if Julia needs rendering
        if (isTimelapseActive) {
             renderJuliaThisFrame = true;
             qualityForJuliaRender = RenderQuality::good;
        }
        else if (mouseMovedInMandelbrotArea && !blockJuliaParameterUpdate && isMouseInMandelbrotUpdateArea)
        {
             renderJuliaThisFrame = true;
             isJuliaVisible = true;
             qualityForJuliaRender = RenderQuality::good; // Quick render for mouse tracking

             // Calculate new Julia parameters based on mouse position relative to Mandelbrot view
             juliaSeedReal = -(mandelbrotFractal.get_x_offset() - (static_cast<double>(mousePosition.x) / mandelbrotFractal.get_zoom_x()));
             juliaSeedImag = -(mandelbrotFractal.get_y_offset() - (static_cast<double>(mousePosition.y) / mandelbrotFractal.get_zoom_y()));
        }
        else { // Check other triggers if not timelapse or mouse move
            if (needsJuliaRender) {
                renderJuliaThisFrame = true;
                qualityForJuliaRender = (currentJuliaQuality == RenderQuality::best) ? RenderQuality::best : RenderQuality::good;
            }
            if (isInteractingJuliaThisFrame) {
                 renderJuliaThisFrame = true;
                 qualityForJuliaRender = RenderQuality::good;
            }
            else if (!renderJuliaThisFrame) { // Check idle improvement
                const float IDLE_TIME_THRESHOLD_JULIA_MS = 500.0f;
                bool needsImprovement = juliaIdleTime.asMilliseconds() > IDLE_TIME_THRESHOLD_JULIA_MS;
                if (needsImprovement && (currentJuliaQuality != RenderQuality::best || previousJuliaQuality != RenderQuality::best))
                {
                    renderJuliaThisFrame = true;
                    qualityForJuliaRender = RenderQuality::best;
                }
            }
        }

        // Perform Julia Render
        if (renderJuliaThisFrame)
        {
            needsJuliaRender = false; // Consume flag
            currentJuliaQuality = qualityForJuliaRender;
            previousJuliaQuality = currentJuliaQuality;

            // juliaRenderClock.restart();

            if (isTimelapseActive) {
                 juliaFractal.update_timelapse(); // Update internal parameters
                 juliaFractal.render(currentJuliaQuality); // Render using internal state
            } else {
                 juliaFractal.render(currentJuliaQuality, juliaSeedReal, juliaSeedImag); // Render with specific parameters
            }

            // Ensure position is correct (might change if texture size changes dynamically)
            juliaFractal.setPosition({ static_cast<float>(windowSize.x - juliaFractal.getTexture().getSize().x), 0.f });

            // auto juliaRenderDuration = juliaRenderClock.getElapsedTime();
            // std::cout << "Julia render time: " << juliaRenderDuration.asMilliseconds() << " ms" << std::endl;
        }


        // ========================================================================
        // --- Final Drawing Stage ---
        // ========================================================================
        window.clear();
        renderTarget.clear();

        // Draw fractals to the render target
        renderTarget.draw(mandelbrotFractal);
        if (isJuliaVisible || isTimelapseActive) {
            renderTarget.draw(juliaFractal);
        }

        renderTarget.display();

        // Draw the render target's texture to the window
        window.draw(sf::Sprite(renderTarget.getTexture()));

        // Draw UI overlays
        window.draw(fpsDisplay);
        window.draw(mandelbrotHardnessDisplay);
        window.draw(juliaHardnessDisplay);
        gui.draw();

        window.display();


        // ========================================================================
        // --- Post-Frame Updates ---
        // ========================================================================

        // Update FPS counter display
        if (fpsUpdateClock.getElapsedTime().asSeconds() > 0.2f) {
            float elapsedSeconds = fpsUpdateClock.restart().asSeconds();
            float fps = (elapsedSeconds > 0) ? static_cast<float>(frameCountForFPS) / elapsedSeconds : 0.0f;
            fpsDisplay.setString("FPS: " + std::to_string(static_cast<int>(fps)));
            mandelbrotHardnessDisplay.setString("M-Hardness: " + std::to_string(mandelbrotFractal.get_hardness_coeff()));
            juliaHardnessDisplay.setString("J-Hardness: " + std::to_string(juliaFractal.get_hardness_coeff()));
            frameCountForFPS = 0;
        }

        // Reset per-frame event flags for the next iteration
        mouseMovedInMandelbrotArea = false;
        isZoomingMandelbrot = false;
        mouseMovedInJuliaArea = false;
        isZoomingJulia = false;

    } // --- End Main Render Loop ---
}

int main() {
    std::thread mainAppThread(main_thread);
    mainAppThread.join();
    return 0;
}