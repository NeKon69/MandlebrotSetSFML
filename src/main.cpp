#include <TGUI/TGUI.hpp>
#include "FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <ctime>

using RenderQuality = render_state;

/*
 * This function contains the main application logic, running in a separate thread.
 * It initializes the window, UI, fractal renderers, and handles the event loop,
 * rendering updates, and state management.
 */
void main_thread() {
    /*
     * Core state variables tracking rendering quality, idle times for automatic
     * quality improvement, and flags indicating whether a re-render is necessary.
     * Render quality switches between 'good' (fast, interactive) and 'best' (slow, detailed).
     */
    float mandelbrotIdleTimeMs = 0.0f;
    sf::Time juliaIdleTime = sf::Time::Zero;
    sf::Clock frameClock;

    RenderQuality currentMandelbrotQuality = RenderQuality::good;
    RenderQuality previousMandelbrotQuality = RenderQuality::good;
    RenderQuality currentJuliaQuality = RenderQuality::good;
    RenderQuality previousJuliaQuality = RenderQuality::good;

    bool needsMandelbrotRender = true;
    bool needsJuliaRender = true;
    bool showMandelbrotIterationLines = false;

    /*
     * Window and rendering target setup using SFML. An off-screen render texture
     * is used to draw the fractals before displaying them on the main window.
     */
    const sf::Vector2u windowSize = { 1920, 1080 };
    sf::RenderWindow window(sf::VideoMode(windowSize), "Fractals");
    sf::RenderTexture renderTarget;
    renderTarget = sf::RenderTexture({windowSize.x, windowSize.y});

    /*
     * Font loading and UI text elements for displaying FPS and fractal hardness.
     * Hardness is likely a metric related to the computational cost of the current view.
     */
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
    juliaHardnessDisplay.setPosition({ static_cast<float>(windowSize.x - 800 + 10), 40.f });
    juliaHardnessDisplay.setString("J-Hardness: 0.0");

    /*
     * State variables related to mouse input, interaction modes (dragging, zooming),
     * Julia set parameters, and application modes like timelapse.
     */
    sf::Vector2i mousePosition;
    double juliaSeedReal = 0.0;
    double juliaSeedImag = 0.0;

    bool isJuliaVisible = false;
    bool isLeftMouseDownMandelbrot = false;
    bool isLeftMouseDownJulia = false;
    bool isTimelapseActive = false;
    bool blockJuliaParameterUpdate = false;

    bool isZoomingMandelbrot = false;
    bool mouseMovedInMandelbrotArea = true;
    bool isZoomingJulia = false;
    bool mouseMovedInJuliaArea = false;

    /*
     * Clocks for timing different operations: FPS updates, render times (though not explicitly displayed),
     * and detecting user inactivity for quality improvements.
     */
    sf::Clock mandelbrotRenderClock;
    sf::Clock juliaRenderClock;
    sf::Clock fpsUpdateClock;
    sf::Clock qualityImprovementClock;

    unsigned int frameCountForFPS = 0;

    /*
     * Fractal object instantiation using the custom CUDA-accelerated FractalBase class.
     * One object for the Mandelbrot set and one for the Julia set.
     */
    using MandelbrotFractal = FractalBase<fractals::mandelbrot>;
    using JuliaFractal = FractalBase<fractals::julia>;

    MandelbrotFractal mandelbrotFractal;
    JuliaFractal juliaFractal;
    juliaFractal.setPosition({ static_cast<float>(windowSize.x - 800), 0.f });

    window.setFramerateLimit(360);

    /*
     * GPU performance estimation to potentially calibrate fractal computation complexity.
     * The measured GFLOPS value is passed to the fractal objects.
     */
    std::cout << "Measuring GFLOPS..." << std::endl;
    float measuredGflops = measureGFLOPS(50000);
    std::cout << "Estimated GFLOPS: " << measuredGflops << std::endl;
    mandelbrotFractal.setMaxComputation(measuredGflops);
    juliaFractal.setMaxComputation(measuredGflops);

    /*
     * TGUI setup: associating the GUI manager with the window and creating widgets
     * (palette selector combobox, HSV offset slider) with their respective callbacks
     * to update fractal parameters and trigger re-renders.
     */
    tgui::Gui gui(window);

    tgui::Slider::Ptr paletteOffsetSlider = tgui::Slider::create();
    paletteOffsetSlider->setMinimum(0);
    paletteOffsetSlider->setMaximum(360);
    paletteOffsetSlider->onValueChange([&](float pos) {
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

    /*
     * The main application loop continues as long as the window is open.
     * It handles events, updates state, determines rendering needs, performs rendering,
     * and draws the final frame.
     */
    while (window.isOpen()) {
        /*
         * Event processing loop: handles all user input and window events.
         * Events are passed to TGUI first. Specific SFML events are handled
         * to manage zooming, panning, dragging, parameter changes, and other interactions.
         * Any significant user interaction resets the 'qualityImprovementClock'.
         */
        while (const auto event = window.pollEvent()) {
             gui.handleEvent(*event);

            if (event->is<sf::Event::MouseWheelScrolled>() ||
                event->is<sf::Event::MouseMoved>() ||
                event->is<sf::Event::KeyPressed>() ||
                event->is<sf::Event::MouseButtonPressed>())
            {
                 qualityImprovementClock.restart();
            }

            if (const auto* mouseMoveEvent = event->getIf<sf::Event::MouseMoved>()) {
                mousePosition = mouseMoveEvent->position;

                bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;
                bool isInJuliaArea = mousePosition.x >= windowSize.x - 800 && mousePosition.x < windowSize.x && mousePosition.y >= 0 && mousePosition.y < 600;

                if (isInMandelbrotArea) {
                    mouseMovedInMandelbrotArea = true;
                    if (showMandelbrotIterationLines) {
                        mandelbrotFractal.drawIterationLines(mousePosition);
                        needsMandelbrotRender = true;
                    }
                    if (mandelbrotFractal.get_is_dragging()) {
                        currentMandelbrotQuality = RenderQuality::good;
                        mandelbrotFractal.dragging(mousePosition);
                        needsMandelbrotRender = true;
                    }
                }
                else if (isInJuliaArea) {
                    mouseMovedInJuliaArea = true;
                     if (juliaFractal.get_is_dragging()) {
                        currentJuliaQuality = RenderQuality::good;
                        juliaFractal.dragging(mousePosition);
                        needsJuliaRender = true;
                     }
                }
            }

            if (event->is<sf::Event::Closed>()) {
                window.close();
                break;
            }

            if (const auto* keyPressEvent = event->getIf<sf::Event::KeyPressed>()) {
                 bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;

                switch(keyPressEvent->scancode) {
                    case sf::Keyboard::Scancode::Escape:
                        window.close();
                        break;

                    case sf::Keyboard::Scancode::R:
                        if (isInMandelbrotArea) {
                            mandelbrotFractal.set_max_iters(mandelbrotFractal.get_max_iters() * 2);
                            needsMandelbrotRender = true;
                            currentMandelbrotQuality = RenderQuality::best;
                        }
                        break;

                    case sf::Keyboard::Scancode::B:
                        if (isInMandelbrotArea) {
                            blockJuliaParameterUpdate = !blockJuliaParameterUpdate;
                             if (!blockJuliaParameterUpdate) needsJuliaRender = true;
                        }
                        break;

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

                            needsMandelbrotRender = true;
                            currentMandelbrotQuality = RenderQuality::good;
                            mandelbrotFractal.move_fractal({int(direction.x), int(direction.y)});
                        }
                        break;

                    case sf::Keyboard::Scancode::T:
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

                    case sf::Keyboard::Scancode::F6:
                    {
                        sf::Texture texture;
                        texture = sf::Texture({mandelbrotFractal.getTexture().getSize().x, mandelbrotFractal.getTexture().getSize().y});
                        if (true) { // This conditional seems redundant
                            texture.update(mandelbrotFractal.getTexture());
                            sf::Image screenshot = texture.copyToImage();

                            auto now = std::chrono::system_clock::now();
                            std::time_t timer = std::chrono::system_clock::to_time_t(now);
                            std::tm time_info;
                            char filenameBuffer[80];
                            #ifdef _WIN32
                            localtime_s(&time_info, &timer);
                            #else
                            localtime_r(&timer, &time_info);
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
                    }
                    default:
                        break;
                }
            }

            if (const auto* mouseWheelEvent = event->getIf<sf::Event::MouseWheelScrolled>()) {
                float delta = mouseWheelEvent->delta;

                bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;
                bool isInJuliaArea = mousePosition.x >= windowSize.x - 800 && mousePosition.x < windowSize.x && mousePosition.y >= 0 && mousePosition.y < 600;

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

            if (const auto* mouseButtonPressedEvent = event->getIf<sf::Event::MouseButtonPressed>()) {
                 bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;
                 bool isInJuliaArea = mousePosition.x >= windowSize.x - 800 && mousePosition.x < windowSize.x && mousePosition.y >= 0 && mousePosition.y < 600;

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
                }
                else if (mouseButtonPressedEvent->button == sf::Mouse::Button::Right) {
                    if (isInMandelbrotArea) {
                        showMandelbrotIterationLines = true;
                        mandelbrotFractal.drawIterationLines(mousePosition);
                        needsMandelbrotRender = true;
                    }
                }
            }

            if (const auto* mouseButtonReleasedEvent = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleasedEvent->button == sf::Mouse::Button::Left) {
                    if (isLeftMouseDownMandelbrot) {
                        isLeftMouseDownMandelbrot = false;
                        currentMandelbrotQuality = RenderQuality::best;
                        mandelbrotFractal.stop_dragging();
                        needsMandelbrotRender = true;
                    }
                    if (isLeftMouseDownJulia) {
                        isLeftMouseDownJulia = false;
                        currentJuliaQuality = RenderQuality::best;
                        juliaFractal.stop_dragging();
                        needsJuliaRender = true;
                    }
                }
                else if (mouseButtonReleasedEvent->button == sf::Mouse::Button::Right) {
                     bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;
                     if (isInMandelbrotArea) {
                         showMandelbrotIterationLines = false;
                         needsMandelbrotRender = true;
                     }
                }
            }

        } // End event loop

        sf::Time deltaTime = frameClock.restart();
        frameCountForFPS++;

        bool isDraggingMandelbrotThisFrame = isLeftMouseDownMandelbrot && mouseMovedInMandelbrotArea;
        bool isZoomingMandelbrotThisFrame = isZoomingMandelbrot;
        bool isInteractingMandelbrotThisFrame = isDraggingMandelbrotThisFrame || isZoomingMandelbrotThisFrame;

        if (isInteractingMandelbrotThisFrame) {
            mandelbrotIdleTimeMs = 0.0f;
        } else {
            mandelbrotIdleTimeMs += deltaTime.asMilliseconds();
        }

        /*
         * Mandelbrot rendering decision logic:
         * Determines if a re-render is needed based on flags ('needsMandelbrotRender'),
         * user interaction ('isInteractingMandelbrotThisFrame'), or idle time exceeding
         * a threshold ('IDLE_TIME_THRESHOLD_MS'). Interaction forces 'good' quality for
         * responsiveness, while idle time triggers 'best' quality for refinement if not
         * already at best quality.
         */
        bool renderMandelbrotThisFrame = false;
        RenderQuality qualityForMandelbrotRender = currentMandelbrotQuality;

        if (needsMandelbrotRender) {
            renderMandelbrotThisFrame = true;
            qualityForMandelbrotRender = (currentMandelbrotQuality == RenderQuality::best) ? RenderQuality::best : RenderQuality::good;
        }
        if (isInteractingMandelbrotThisFrame) {
            renderMandelbrotThisFrame = true;
            qualityForMandelbrotRender = RenderQuality::good;
        }
        else if (!renderMandelbrotThisFrame) {
            const float IDLE_TIME_THRESHOLD_MS = 500.0f;
            bool needsQualityImprovement = mandelbrotIdleTimeMs > IDLE_TIME_THRESHOLD_MS;
            if (needsQualityImprovement && (currentMandelbrotQuality != RenderQuality::best || previousMandelbrotQuality != RenderQuality::best)) {
                renderMandelbrotThisFrame = true;
                qualityForMandelbrotRender = RenderQuality::best;
            }
        }

        if (renderMandelbrotThisFrame)
        {
            needsMandelbrotRender = false;
            currentMandelbrotQuality = qualityForMandelbrotRender;
            previousMandelbrotQuality = currentMandelbrotQuality;

            mandelbrotRenderClock.restart();
            mandelbrotFractal.render(currentMandelbrotQuality);
        }

        /*
         * Julia rendering decision logic:
         * Similar to Mandelbrot, but also considers timelapse mode ('isTimelapseActive')
         * and whether the mouse is hovering over the Mandelbrot set to update Julia parameters
         * ('mouseMovedInMandelbrotArea', '!blockJuliaParameterUpdate'). Timelapse and hover updates
         * typically use 'good' quality. Idle improvement logic is also present, similar to Mandelbrot,
         * but avoids triggering if the mouse is over the Mandelbrot area causing parameter updates.
         */
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
        bool isMouseInMandelbrotUpdateArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;


        if (isTimelapseActive) {
             renderJuliaThisFrame = true;
             qualityForJuliaRender = RenderQuality::good;
        }
        else if (mouseMovedInMandelbrotArea && !blockJuliaParameterUpdate && isMouseInMandelbrotUpdateArea)
        {
             renderJuliaThisFrame = true;
             isJuliaVisible = true;
             qualityForJuliaRender = RenderQuality::good;

             juliaSeedReal = -(mandelbrotFractal.get_x_offset() - (static_cast<double>(mousePosition.x) / mandelbrotFractal.get_zoom_x()));
             juliaSeedImag = -(mandelbrotFractal.get_y_offset() - (static_cast<double>(mousePosition.y) / mandelbrotFractal.get_zoom_y()));
        }
        else {
            if (needsJuliaRender) {
                renderJuliaThisFrame = true;
                qualityForJuliaRender = (currentJuliaQuality == RenderQuality::best) ? RenderQuality::best : RenderQuality::good;
            }
            if (isInteractingJuliaThisFrame) {
                 renderJuliaThisFrame = true;
                 qualityForJuliaRender = RenderQuality::good;
            }
            else if (!renderJuliaThisFrame) {
                const float IDLE_TIME_THRESHOLD_JULIA_MS = 500.0f;
                bool needsImprovement = juliaIdleTime.asMilliseconds() > IDLE_TIME_THRESHOLD_JULIA_MS;
                bool isMouseOverMandelbrot = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;

                if (needsImprovement && !isMouseOverMandelbrot && (currentJuliaQuality != RenderQuality::best || previousJuliaQuality != RenderQuality::best))
                {
                    renderJuliaThisFrame = true;
                    qualityForJuliaRender = RenderQuality::best;
                }
            }
        }

        if (renderJuliaThisFrame)
        {
            needsJuliaRender = false;
            currentJuliaQuality = qualityForJuliaRender;
            previousJuliaQuality = currentJuliaQuality;

            juliaRenderClock.restart();
            if (isTimelapseActive) {
                 /*
                  * Updates Julia parameters based on the timelapse logic.
                  * Calling it twice might be intentional to advance the animation faster
                  * or could be a potential area for review.
                  */
                 juliaFractal.update_timelapse();
                 juliaFractal.update_timelapse();
            } else {
                 juliaFractal.render(currentJuliaQuality, juliaSeedReal, juliaSeedImag);
            }
            juliaFractal.setPosition({ static_cast<float>(windowSize.x - 800), 0.f });
        }

        /*
         * Drawing phase: Clears buffers, draws fractals to the render target,
         * draws the render target to the window, then draws UI elements (text, TGUI) on top.
         */
        window.clear();
        renderTarget.clear();

        renderTarget.draw(mandelbrotFractal);
        if (isJuliaVisible || isTimelapseActive) {
            renderTarget.draw(juliaFractal);
        }

        renderTarget.display();

        window.draw(sf::Sprite(renderTarget.getTexture()));

        window.draw(fpsDisplay);
        window.draw(mandelbrotHardnessDisplay);
        window.draw(juliaHardnessDisplay);
        gui.draw();

        window.display();

        /*
         * Periodic update of FPS and hardness display based on a timer.
         */
        if (fpsUpdateClock.getElapsedTime().asSeconds() > 0.2f) {
            float elapsedSeconds = fpsUpdateClock.restart().asSeconds();
            float fps = frameCountForFPS / elapsedSeconds;
            fpsDisplay.setString("FPS: " + std::to_string(fps));
            mandelbrotHardnessDisplay.setString("M-Hardness: " + std::to_string(mandelbrotFractal.get_hardness_coeff()));
            juliaHardnessDisplay.setString("J-Hardness: " + std::to_string(juliaFractal.get_hardness_coeff()));
            frameCountForFPS = 0;
        }

        /*
         * Reset per-frame boolean flags related to mouse movement and zooming.
         */
        mouseMovedInMandelbrotArea = false;
        isZoomingMandelbrot = false;
        mouseMovedInJuliaArea = false;
        isZoomingJulia = false;

    } // End main loop
}

int main() {
    std::thread mainAppThread(main_thread);
    mainAppThread.join();
    return 0;
}