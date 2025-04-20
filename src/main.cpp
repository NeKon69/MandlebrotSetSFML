#include <TGUI/TGUI.hpp>
#include "FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <ctime>

using RenderQuality = render_state;

void ComboBoxCreator(tgui::ComboBox::Ptr& comboBox, const std::vector<std::string>& items, const std::string& defaultItem) {
    for (const auto& item : items) {
        comboBox->addItem(item);
    }
    comboBox->setSelectedItem(defaultItem);
}

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
    const sf::Vector2u windowSize = { 2560, 1440 };
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
    fpsDisplay.setFillColor(sf::Color::Black);
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
    bool progressiveAnimation = false;

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

    /*
     * GPU performance estimation to potentially calibrate fractal computation complexity.
     * The measured GFLOPS value is passed to the fractal objects.
     */
    std::cout << "Measuring GFLOPS..." << std::endl;
    float measuredGflops = measureGFLOPS(50000);
    std::cout << "Estimated GFLOPS: " << measuredGflops << std::endl;
    mandelbrotFractal.setMaxComputation(measuredGflops);
    juliaFractal.setMaxComputation(measuredGflops);

    /*Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)
     * TGUI setup: associating the GUI manager with the window and creating widgets
     * (palette selector combobox, HSV offset slider) with their respective callbacks
     * to update fractal parameters and trigger re-renders.
     */
    tgui::Gui gui(window);

    const std::vector<std::string> paletteNames = {
            "HSV", "BlackOWhite", "Basic", "OscillatingGrayscale", "Fire", "Pastel",
            "Interpolated", "CyclicHSV", "FractalPattern", "PerlinNoise", "Water",
            "Sunset", "DeepSpace", "Physchodelic", "IceCave", "ElectricNebula",
            "AccretionDisk"};

    // --- Common layout values ---
    const float controlPadding = 10.f;
    const float labelControlVPadding = 4.f;
    const float controlGroupXOffsetMandelbrot = 10.f;
    const float controlWidth = 200.f;
    const float iterEditBoxWidth = 65.f;
    const float iterSliderWidth = controlWidth - iterEditBoxWidth - controlPadding;
    const tgui::Layout controlGroupXOffsetJulia = {"parent.width - " + std::to_string(controlWidth + controlPadding)};
    const unsigned int minIterations = 50;
    const unsigned int maxIterations = 10000;
    constexpr unsigned int DELIMETER_FOR_ITERATION_UPDATE = 200;
    constexpr unsigned int UPDATE_FRAMES_OF_ANIMATION = 60;
    sf::Clock timer_update_iteration;

    // --- Mandelbrot Controls ---
    sf::Vector2i initialMandelbrotRes = mandelbrotFractal.get_resolution();
    float mandelbrotControlsStartY = static_cast<float>(initialMandelbrotRes.y) + controlPadding;

    tgui::Label::Ptr mandelbrotLabel = tgui::Label::create("Mandelbrot Settings");
    mandelbrotLabel->setPosition(controlGroupXOffsetMandelbrot, mandelbrotControlsStartY);
    mandelbrotLabel->setTextSize(16);
    gui.add(mandelbrotLabel);

    tgui::ComboBox::Ptr paletteSelectorMandelbrot = tgui::ComboBox::create();
    paletteSelectorMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(mandelbrotLabel) + controlPadding);
    paletteSelectorMandelbrot->setSize({controlWidth, 24});
    ComboBoxCreator(paletteSelectorMandelbrot, paletteNames, "HSV");
    paletteSelectorMandelbrot->onItemSelect([&](const tgui::String& str) {
        mandelbrotFractal.setPallete(str.toStdString());
        needsMandelbrotRender = true;
    });
    gui.add(paletteSelectorMandelbrot);

    tgui::Slider::Ptr paletteOffsetSliderMandelbrot = tgui::Slider::create();
    paletteOffsetSliderMandelbrot->setMinimum(0);
    paletteOffsetSliderMandelbrot->setMaximum(360);
    paletteOffsetSliderMandelbrot->setValue(0);
    paletteOffsetSliderMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(paletteSelectorMandelbrot) + controlPadding);
    paletteOffsetSliderMandelbrot->setSize({controlWidth, 18});
    paletteOffsetSliderMandelbrot->onValueChange([&](float pos) {
        if (mandelbrotFractal.getPallete() == Palletes::HSV || mandelbrotFractal.getPallete() == Palletes::CyclicHSV) {
            mandelbrotFractal.SetDegreesOffsetForHSV(static_cast<int>(pos));
            needsMandelbrotRender = true;
        }
    });
    gui.add(paletteOffsetSliderMandelbrot);

    tgui::EditBox::Ptr resolutionPickerMandelbrot = tgui::EditBox::create();
    resolutionPickerMandelbrot->setText(std::to_string(initialMandelbrotRes.x) + "x" + std::to_string(initialMandelbrotRes.y));
    resolutionPickerMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(paletteOffsetSliderMandelbrot) + controlPadding);
    resolutionPickerMandelbrot->setSize({controlWidth, 24}); // Use full control width
    resolutionPickerMandelbrot->onTextChange([&](const tgui::String& str) {
        std::string text = str.toStdString();
        size_t x_pos = text.find('x');
        resolutionPickerMandelbrot->getRenderer()->setBorderColor(sf::Color::Black);
        if (x_pos != std::string::npos && x_pos > 0 && text.length() > x_pos + 1) {
            try {
                std::string width_str = text.substr(0, x_pos);
                std::string height_str = text.substr(x_pos + 1);
                int width = std::stoi(width_str);
                int height = std::stoi(height_str);
                if(width >= 100 && height >= 100 && width <= 8192 && height <= 8192) {
                    mandelbrotFractal.set_resolution({width, height});
                    needsMandelbrotRender = true;
                } else {
                    resolutionPickerMandelbrot->getRenderer()->setBorderColor(sf::Color::Yellow);
                }
            } catch (const std::invalid_argument&) {
                resolutionPickerMandelbrot->getRenderer()->setBorderColor(sf::Color::Red);
            } catch (const std::out_of_range&) {
                resolutionPickerMandelbrot->getRenderer()->setBorderColor(sf::Color::Red);
            }
        } else if (!text.empty()) {
            resolutionPickerMandelbrot->getRenderer()->setBorderColor(sf::Color::Red);
        }
    });
    gui.add(resolutionPickerMandelbrot);

    // --- Max Iterations Control (Mandelbrot) ---
    tgui::Label::Ptr maxIterLabelMandelbrot = tgui::Label::create("Max Iterations:");
    maxIterLabelMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(resolutionPickerMandelbrot) + controlPadding * 1.5f); // Add a bit more space
    gui.add(maxIterLabelMandelbrot);
    unsigned int currentMandelbrotIters = std::max(minIterations, std::min(maxIterations, mandelbrotFractal.get_max_iters()));
    mandelbrotFractal.set_max_iters(currentMandelbrotIters);

    tgui::EditBox::Ptr maxIterEditBoxMandelbrot = tgui::EditBox::create();
    maxIterEditBoxMandelbrot->setDefaultText(std::to_string(currentMandelbrotIters));
    maxIterEditBoxMandelbrot->setText(std::to_string(currentMandelbrotIters));
    maxIterEditBoxMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(maxIterLabelMandelbrot) + labelControlVPadding);
    maxIterEditBoxMandelbrot->setSize({iterEditBoxWidth, 24});
    maxIterEditBoxMandelbrot->setInputValidator(tgui::EditBox::Validator::UInt);
    gui.add(maxIterEditBoxMandelbrot);

    tgui::Slider::Ptr maxIterSliderMandelbrot = tgui::Slider::create();
    maxIterSliderMandelbrot->setMinimum(static_cast<float>(minIterations));
    maxIterSliderMandelbrot->setMaximum(static_cast<float>(maxIterations));
    maxIterSliderMandelbrot->setValue(static_cast<float>(currentMandelbrotIters));
    maxIterSliderMandelbrot->setPosition(tgui::bindRight(maxIterEditBoxMandelbrot) + controlPadding, tgui::bindTop(maxIterEditBoxMandelbrot) + (tgui::bindHeight(maxIterEditBoxMandelbrot) - tgui::bindHeight(maxIterSliderMandelbrot)) / 2.f);
    maxIterSliderMandelbrot->setSize({iterSliderWidth, 18});
    gui.add(maxIterSliderMandelbrot);

    maxIterSliderMandelbrot->onValueChange([&, maxIterEditBoxMandelbrot](float value) {
        unsigned int intValue = static_cast<unsigned int>(value);
        intValue = std::max(minIterations, std::min(maxIterations, intValue));
        mandelbrotFractal.set_max_iters(intValue);
        if (maxIterEditBoxMandelbrot) {
            maxIterEditBoxMandelbrot->setText(std::to_string(intValue));
            maxIterEditBoxMandelbrot->getRenderer()->setBorderColor("black");
        }
        needsMandelbrotRender = true;
    });

    auto updateMandelbrotIterationsFromEditBox = [&, maxIterSliderMandelbrot, maxIterEditBoxMandelbrot]() {
        unsigned int currentValue = mandelbrotFractal.get_max_iters();
        try {
            if (maxIterEditBoxMandelbrot && !maxIterEditBoxMandelbrot->getText().empty()) {
                long long parsedValue = std::stoll(maxIterEditBoxMandelbrot->getText().toStdString());
                if (parsedValue >= minIterations && parsedValue <= maxIterations) {
                    currentValue = static_cast<unsigned int>(parsedValue);
                    mandelbrotFractal.set_max_iters(currentValue);
                    if(maxIterSliderMandelbrot) {
                        maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue));
                    }
                    maxIterEditBoxMandelbrot->getRenderer()->setBorderColor("black");
                    needsMandelbrotRender = true;
                } else {
                    currentValue = mandelbrotFractal.get_max_iters();
                    maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                    if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); } // Also reset slider
                    maxIterEditBoxMandelbrot->getRenderer()->setBorderColor("orange");
                }
            } else if (maxIterEditBoxMandelbrot) {
                currentValue = mandelbrotFractal.get_max_iters();
                maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxMandelbrot->getRenderer()->setBorderColor("black");
            }
        } catch (const std::invalid_argument&) {
            if (maxIterEditBoxMandelbrot) {
                currentValue = mandelbrotFractal.get_max_iters();
                maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxMandelbrot->getRenderer()->setBorderColor("red");
            }
        } catch (const std::out_of_range&) {
            if (maxIterEditBoxMandelbrot) {
                currentValue = mandelbrotFractal.get_max_iters();
                maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxMandelbrot->getRenderer()->setBorderColor("red");
            }
        }
    };
    maxIterEditBoxMandelbrot->onReturnOrUnfocus(updateMandelbrotIterationsFromEditBox);

    tgui::Button::Ptr ResetMandelbrot = tgui::Button::create("Reset Mandelbrot");
    ResetMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(maxIterEditBoxMandelbrot) + controlPadding);
    ResetMandelbrot->setSize({controlWidth, 24});
    ResetMandelbrot->onPress([&]() {
        mandelbrotFractal.reset();
        needsMandelbrotRender = true;
    });
    gui.add(ResetMandelbrot);

    // --- Julia Controls ---

    sf::Vector2i initialJuliaRes = juliaFractal.get_resolution();
    float juliaControlsStartY = static_cast<float>(initialJuliaRes.y) + controlPadding;

    tgui::Label::Ptr juliaLabel = tgui::Label::create("Julia Settings");
    juliaLabel->setPosition(controlGroupXOffsetJulia, juliaControlsStartY);
    juliaLabel->setTextSize(16);
    gui.add(juliaLabel);

    tgui::ComboBox::Ptr paletteSelectorJulia = tgui::ComboBox::create();
    paletteSelectorJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(juliaLabel) + controlPadding);
    paletteSelectorJulia->setSize({controlWidth, 24});
    ComboBoxCreator(paletteSelectorJulia, paletteNames, "HSV");
    paletteSelectorJulia->onItemSelect([&](const tgui::String& str) {
        juliaFractal.setPallete(str.toStdString());
        needsJuliaRender = true;
    });
    gui.add(paletteSelectorJulia);

    tgui::Slider::Ptr paletteOffsetSliderJulia = tgui::Slider::create();
    paletteOffsetSliderJulia->setMinimum(0);
    paletteOffsetSliderJulia->setMaximum(360);
    paletteOffsetSliderJulia->setValue(0);
    paletteOffsetSliderJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(paletteSelectorJulia) + controlPadding);
    paletteOffsetSliderJulia->setSize({controlWidth, 18});
    paletteOffsetSliderJulia->onValueChange([&](float pos) {
        if (juliaFractal.getPallete() == Palletes::HSV || juliaFractal.getPallete() == Palletes::CyclicHSV) {
            juliaFractal.SetDegreesOffsetForHSV(static_cast<int>(pos));
            needsJuliaRender = true;
        }
    });
    gui.add(paletteOffsetSliderJulia);


    tgui::EditBox::Ptr resolutionPickerJulia = tgui::EditBox::create();
    resolutionPickerJulia->setText(std::to_string(initialJuliaRes.x) + "x" + std::to_string(initialJuliaRes.y));
    resolutionPickerJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(paletteOffsetSliderJulia) + controlPadding);
    resolutionPickerJulia->setSize({controlWidth, 24});
    resolutionPickerJulia->onTextChange([&](const tgui::String& str) {
        std::string text = str.toStdString();
        size_t x_pos = text.find('x');
        resolutionPickerJulia->getRenderer()->setBorderColor(sf::Color::Black);
        if (x_pos != std::string::npos && x_pos > 0 && text.length() > x_pos + 1) {
            try {
                std::string width_str = text.substr(0, x_pos);
                std::string height_str = text.substr(x_pos + 1);
                int width = std::stoi(width_str);
                int height = std::stoi(height_str);
                if(width >= 100 && height >= 100 && width <= 8192 && height <= 8192) {
                    juliaFractal.set_resolution({width, height});
                    needsJuliaRender = true;
                } else {
                    resolutionPickerJulia->getRenderer()->setBorderColor(sf::Color::Yellow);
                }
            } catch (const std::invalid_argument&) {
                resolutionPickerJulia->getRenderer()->setBorderColor(sf::Color::Red);
            } catch (const std::out_of_range&) {
                resolutionPickerJulia->getRenderer()->setBorderColor(sf::Color::Red);
            }
        } else if (!text.empty()) {
            resolutionPickerJulia->getRenderer()->setBorderColor(sf::Color::Red);
        }
    });
    gui.add(resolutionPickerJulia);

    // --- Max Iterations Control (Julia) ---
    tgui::Label::Ptr maxIterLabelJulia = tgui::Label::create("Max Iterations:");
    maxIterLabelJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(resolutionPickerJulia) + controlPadding * 1.5f);
    gui.add(maxIterLabelJulia);
    unsigned int currentJuliaIters = std::max(minIterations, std::min(maxIterations, juliaFractal.get_max_iters()));
    juliaFractal.set_max_iters(currentJuliaIters);

    tgui::EditBox::Ptr maxIterEditBoxJulia = tgui::EditBox::create();
    maxIterEditBoxJulia->setDefaultText(std::to_string(currentJuliaIters));
    maxIterEditBoxJulia->setText(std::to_string(currentJuliaIters));
    maxIterEditBoxJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(maxIterLabelJulia) + labelControlVPadding);
    maxIterEditBoxJulia->setSize({iterEditBoxWidth, 24});
    maxIterEditBoxJulia->setInputValidator(tgui::EditBox::Validator::UInt);
    gui.add(maxIterEditBoxJulia);

    tgui::Slider::Ptr maxIterSliderJulia = tgui::Slider::create();
    maxIterSliderJulia->setMinimum(static_cast<float>(minIterations));
    maxIterSliderJulia->setMaximum(static_cast<float>(maxIterations));
    maxIterSliderJulia->setValue(static_cast<float>(currentJuliaIters));
    maxIterSliderJulia->setPosition(tgui::bindRight(maxIterEditBoxJulia) + controlPadding, tgui::bindTop(maxIterEditBoxJulia) + (tgui::bindHeight(maxIterEditBoxJulia) - tgui::bindHeight(maxIterSliderJulia)) / 2.f);
    maxIterSliderJulia->setSize({iterSliderWidth, 18});
    gui.add(maxIterSliderJulia);

    maxIterSliderJulia->onValueChange([&, maxIterEditBoxJulia](float value) { // Capture edit box pointer
        unsigned int intValue = static_cast<unsigned int>(value);
        intValue = std::max(minIterations, std::min(maxIterations, intValue)); // Clamp
        juliaFractal.set_max_iters(intValue);
        if (maxIterEditBoxJulia) {
            maxIterEditBoxJulia->setText(std::to_string(intValue));
            maxIterEditBoxJulia->getRenderer()->setBorderColor("black");
        }
        needsJuliaRender = true;
    });

    auto updateJuliaIterationsFromEditBox = [&, maxIterSliderJulia, maxIterEditBoxJulia]() {
        unsigned int currentValue = juliaFractal.get_max_iters();
        try {
            if (maxIterEditBoxJulia && !maxIterEditBoxJulia->getText().empty()) {
                long long parsedValue = std::stoll(maxIterEditBoxJulia->getText().toStdString());
                if (parsedValue >= minIterations && parsedValue <= maxIterations) {
                    currentValue = static_cast<unsigned int>(parsedValue);
                    juliaFractal.set_max_iters(currentValue);
                    if(maxIterSliderJulia) {
                        maxIterSliderJulia->setValue(static_cast<float>(currentValue));
                    }
                    maxIterEditBoxJulia->getRenderer()->setBorderColor("black");
                    needsJuliaRender = true;
                } else {
                    currentValue = juliaFractal.get_max_iters();
                    maxIterEditBoxJulia->setText(std::to_string(currentValue));
                    if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                    maxIterEditBoxJulia->getRenderer()->setBorderColor("orange");
                }
            } else if (maxIterEditBoxJulia) {
                currentValue = juliaFractal.get_max_iters();
                maxIterEditBoxJulia->setText(std::to_string(currentValue));
                if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxJulia->getRenderer()->setBorderColor("black");
            }
        } catch (const std::invalid_argument&) {
            if (maxIterEditBoxJulia) {
                currentValue = juliaFractal.get_max_iters();
                maxIterEditBoxJulia->setText(std::to_string(currentValue));
                if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxJulia->getRenderer()->setBorderColor("red");
            }
        } catch (const std::out_of_range&) {
            if (maxIterEditBoxJulia) {
                currentValue = juliaFractal.get_max_iters();
                maxIterEditBoxJulia->setText(std::to_string(currentValue));
                if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxJulia->getRenderer()->setBorderColor("red");
            }
        }
    };
    maxIterEditBoxJulia->onReturnOrUnfocus(updateJuliaIterationsFromEditBox);

    tgui::Button::Ptr ResetJulia = tgui::Button::create("Reset Julia");
    ResetJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(maxIterEditBoxJulia) + controlPadding);
    ResetJulia->setSize({controlWidth, 24});
    ResetJulia->onPress([&]() {
        juliaFractal.reset();
        needsJuliaRender = true;
    });
    gui.add(ResetJulia);

    tgui::ToggleButton::Ptr UpdateIterationsJulia = tgui::ToggleButton::create("Start Cool Animation");
    unsigned int startingIterations = 300;
    UpdateIterationsJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(ResetJulia) + controlPadding);
    UpdateIterationsJulia->setSize({controlWidth, 24});
    UpdateIterationsJulia->onToggle([&](){
        if(progressiveAnimation)
            juliaFractal.set_max_iters(startingIterations);
        else
            startingIterations = juliaFractal.get_max_iters();
        progressiveAnimation = !progressiveAnimation;
    });
    gui.add(UpdateIterationsJulia);
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
                    }
                    if (mandelbrotFractal.get_is_dragging()) {
                        currentMandelbrotQuality = RenderQuality::good;
                        mandelbrotFractal.dragging(mousePosition);
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
                    case sf::Keyboard::Scancode::F11:
                        const sf::Vector2i fullScreenResolution = {2560, 1440};
                        mandelbrotFractal.set_resolution({int(fullScreenResolution.x * mandelbrotFractal.get_resolution().x / window.getSize().x), int(fullScreenResolution.y * mandelbrotFractal.get_resolution().y / window.getSize().y)});
                        juliaFractal.set_resolution({int(fullScreenResolution.x * juliaFractal.get_resolution().x / window.getSize().x), int(fullScreenResolution.y * juliaFractal.get_resolution().y / window.getSize().y)});
                        window.setSize({(unsigned int)(fullScreenResolution.x), (unsigned int)(fullScreenResolution.y)});
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
                }
                else if (isInJuliaArea) {
                    isZoomingJulia = true;
                    currentJuliaQuality = RenderQuality::good;
                    juliaFractal.handleZoom(delta, {int(mousePosition.x - windowSize.x + 800), mousePosition.y});
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
                        mandelbrotFractal.stop_dragging();
                    }
                    if (isLeftMouseDownJulia) {
                        isLeftMouseDownJulia = false;
                        juliaFractal.stop_dragging();
                    }
                }
                else if (mouseButtonReleasedEvent->button == sf::Mouse::Button::Right) {
                     bool isInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < 800 && mousePosition.y >= 0 && mousePosition.y < 600;
                     if (isInMandelbrotArea) {
                         showMandelbrotIterationLines = false;
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
            qualityForMandelbrotRender = RenderQuality::good;
        }
        if (isInteractingMandelbrotThisFrame) {
            renderMandelbrotThisFrame = true;
            qualityForMandelbrotRender = RenderQuality::good;
        }
        else if (!renderMandelbrotThisFrame) {
            const float IDLE_TIME_THRESHOLD_MS = 500.0f;
            bool needsQualityImprovement = mandelbrotIdleTimeMs > IDLE_TIME_THRESHOLD_MS / 10;
            if (needsQualityImprovement && currentMandelbrotQuality != RenderQuality::best && previousMandelbrotQuality != RenderQuality::best){
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
            std::cout << "Rendering Mandelbrot with quality: " << (currentMandelbrotQuality == render_state::good? "Good" : "Best") << std::endl;
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
        else if (progressiveAnimation){
            renderJuliaThisFrame = true;
            if (timer_update_iteration.getElapsedTime().asMilliseconds() > 1000 / UPDATE_FRAMES_OF_ANIMATION) {
                timer_update_iteration.restart();
                unsigned int curr_iters = juliaFractal.get_max_iters();
                juliaFractal.set_max_iters(curr_iters + curr_iters / DELIMETER_FOR_ITERATION_UPDATE);
            }

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
        window.clear(sf::Color::Black);
        renderTarget.clear(sf::Color::Black);

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