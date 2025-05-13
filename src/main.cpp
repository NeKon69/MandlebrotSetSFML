#include <TGUI/TGUI.hpp>
#include "ClassImplementation/FractalClass.cuh"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <ctime>

/// Alias for the render state enum, controlling the quality
/// of the fractal rendering (e.g., 'good' for interactive,
/// 'best' for final detail).
using RenderQuality = render_state;

/// Macros to calculate scaled sizes based on the current window
/// resolution relative to a reference resolution (2560x1440).
/// Used to make UI elements scale with the window.
#define calculate_size_y(based) \
  (based / (1440.f / window.getSize().y))

#define calculate_size_x(based) \
  (based / (2560.f / window.getSize().x))

/// Helper function to populate a TGUI ComboBox with a list of items
/// and set a default selected item.
void ComboBoxCreator(tgui::ComboBox::Ptr& comboBox, const std::vector<std::string>& items, const std::string& defaultItem) {
    for (const auto& item : items) {
        comboBox->addItem(item);
    }
    comboBox->setSelectedItem(defaultItem);
}

/// Calculates the global screen bounds (position and size) for a TGUI panel.
/// This is used to determine if the mouse is within a UI element's area.
sf::FloatRect calculateManualGlobalBounds(const tgui::Panel::Ptr& object) {
    if (!object) {
        return sf::FloatRect{};
    }
    sf::Vector2f globalTopLeft = object->getPosition();
    sf::Vector2f size = object->getSize();
    return sf::FloatRect{globalTopLeft, size};
}

/// Checks if the mouse cursor is currently positioned inside any
/// of the provided TGUI panel objects. Used to prevent fractal
/// interaction when the mouse is over UI controls.
bool isMouseInsideAnyLabelManual(sf::Vector2i mousePos, const std::initializer_list<tgui::Panel::Ptr>& objects) {
    auto mousePosF = static_cast<sf::Vector2f>(mousePos);

    return std::ranges::any_of(objects, [&](const tgui::Panel::Ptr& object) {
        if (!object) return false;
        sf::FloatRect globalBounds = calculateManualGlobalBounds(object);
        return globalBounds.contains(mousePosF);
    });
}

/// The main application logic.
/// Initializes the window, UI, fractal renderers, and handles the event loop,
/// rendering updates, and state management.
int main() {
    /// Core application state variables.
    /// mandelbrotIdleTimeMs, juliaIdleTime: Track time since last interaction
    /// to trigger quality improvements.
    /// current/previousQuality: Manage rendering quality levels (good/best).
    /// needsRender: Flags to force a re-render.
    /// showMandelbrotIterationLines: Toggle for debugging visualization.
    /// updateIterationsDynamically: Flags for automatic iteration adjustment based on zoom.
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
    bool updateIterationsDynamicallyMandelbrot = false;
    bool updateIterationsDynamicallyJulia = false;

    /// SFML window and off-screen render texture setup.
    /// The render texture is used to draw fractal content before
    /// presenting it to the main window.
    sf::VideoMode windowSize = sf::VideoMode::getDesktopMode();
    sf::RenderWindow window(windowSize, "Fractals", sf::Style::None, sf::State::Windowed);
    sf::RenderTexture renderTarget;
    renderTarget = sf::RenderTexture(windowSize.size);
    sf::Vector2u oldWindowSize = window.getSize();

    /// Instantiation of the custom fractal objects.
    /// These classes handle the fractal computation and rendering,
    using MandelbrotFractal = FractalBase<fractals::mandelbrot>;
    using JuliaFractal = FractalBase<fractals::julia>;

    MandelbrotFractal mandelbrotFractal;
    JuliaFractal juliaFractal;
    juliaFractal.setPosition({ static_cast<float>(window.getSize().x - 800), 0.f });

    /// Font loading and SFML Text objects for displaying information
    /// like FPS and fractal "hardness" (a measure of computational cost).
    sf::Font mainFont;
    if (!mainFont.openFromFile(R"(fonts/LiberationMono-Bold.ttf)")) {
        std::cerr << "Error loading font!" << std::endl;
        return -1;
    }
    sf::Text fpsDisplay(mainFont);
    fpsDisplay.setPosition({ 10, 10 });
    fpsDisplay.setString("FPS: 0");
    sf::Text mandelbrotHardnessDisplay(mainFont);
    mandelbrotHardnessDisplay.setPosition({ 10, 40 });
    mandelbrotHardnessDisplay.setString("M-Hardness: 0.0");
    sf::Text juliaHardnessDisplay(mainFont);
    juliaHardnessDisplay.setPosition({ static_cast<float>(window.getSize().x - juliaFractal.get_resolution().x + 10), 40.f });
    juliaHardnessDisplay.setString("J-Hardness: 0.0");

    /// State variables related to user interaction and application modes.
    /// mousePosition: Current mouse coordinates.
    /// juliaSeedReal/Imag: Parameters for the Julia set, often linked to
    /// the mouse position over the Mandelbrot set.
    /// isJuliaVisible: Toggle for showing/hiding the Julia set.
    /// isLeftMouseDown: Flags for tracking dragging state.
    /// isTimelapseActive: Flag for the Julia set animation mode.
    /// blockJuliaParameterUpdate: Prevents mouse position from updating Julia parameters.
    /// progressiveAnimation: Flag for the Julia iteration animation.
    /// isZooming: Flags for tracking zoom state.
    /// mouseMovedInArea: Flags to detect mouse movement within fractal areas.
    /// IsInArea: Flags to check if the mouse is currently within a fractal's display area.
    sf::Vector2i mousePosition;
    double juliaSeedReal = 0.0;
    double juliaSeedImag = 0.0;

    bool isJuliaVisible = true;
    bool isLeftMouseDownMandelbrot = false;
    bool isLeftMouseDownJulia = false;
    bool isTimelapseActive = false;
    bool blockJuliaParameterUpdate = false;
    bool progressiveAnimation = false;

    bool isZoomingMandelbrot = false;
    bool mouseMovedInMandelbrotArea = true;
    bool isZoomingJulia = false;
    bool mouseMovedInJuliaArea = false;

    bool IsInMandelbrotArea = false;
    bool IsInJuliaArea = false;

    bool updateScreen = true;


    /// SFML Clocks for timing various application aspects.
    /// renderClock: Measures fractal rendering time (not currently displayed).
    /// fpsUpdateClock: Controls the frequency of FPS display updates.
    /// qualityImprovementClock: Tracks idle time for triggering 'best' quality renders.
    sf::Clock mandelbrotRenderClock;
    sf::Clock juliaRenderClock;
    sf::Clock fpsUpdateClock;
    sf::Clock qualityImprovementClock;

    unsigned int frameCountForFPS = 0;

    /// GPU performance estimation.
    /// Measures GFLOPS/GDFLOPS to potentially calibrate fractal
    /// computation complexity or dynamic iteration limits.
    if(mandelbrotFractal.get_isCudaAvailable()) {
        std::cout << "Measuring GFLOPS..." << std::endl;
        float measuredGflops = measureGFLOPS(50000);
        std::cout << "Estimated GFLOPS: " << measuredGflops << std::endl;
        float measuredGDflops = measureGDFLOPS(50000);
        mandelbrotFractal.setMaxComputation(measuredGflops, measuredGDflops);
        juliaFractal.setMaxComputation(measuredGflops, measuredGDflops);
    }
    else {
        std::cerr << "Can't measure GFLOPS without nvidia GPU\n";
    }

    /// TGUI (Tiny GUI) setup.
    /// Initializes the GUI manager and prepares it to handle UI elements.
    tgui::Gui gui(window);

    const std::vector<std::string> paletteNames = {
            "HSV", "BlackOWhite", "Basic", "OscillatingGrayscale", "Fire", "Pastel",
            "Interpolated", "CyclicHSV", "FractalPattern", "PerlinNoise", "Water",
            "Sunset", "DeepSpace", "Physchodelic", "IceCave", "ElectricNebula",
            "AccretionDisk", "Random"};

    const std::string resolution = std::to_string(int(window.getSize().x / 2)) + "x" + std::to_string(int(window.getSize().y));
    const float controlPadding = calculate_size_y(10.f);
    const float labelControlVPadding = calculate_size_y(4.f);
    const float controlGroupXOffsetMandelbrot = calculate_size_x(10.f);
    const float controlWidth = calculate_size_y(200.f);
    const float iterEditBoxWidth = calculate_size_y(65.f);
    const float iterSliderWidth = controlWidth - iterEditBoxWidth - controlPadding;
    const tgui::Layout controlGroupXOffsetJulia = {"parent.width - " + std::to_string(controlWidth + controlPadding)};
    const unsigned int minIterations = 50;
    const unsigned int maxIterations = 10000;
    constexpr unsigned int DELIMETER_FOR_ITERATION_UPDATE = 200;
    constexpr unsigned int UPDATE_FRAMES_OF_ANIMATION = 60;
    sf::Clock timer_update_iteration;

    /// Mandelbrot control panel and widgets.
    /// Sets up UI elements for configuring the Mandelbrot fractal,
    /// including palette selection, HSV offset, resolution, max iterations,
    /// reset button, progressive iteration checkbox, and compilation controls.
    sf::Vector2i initialMandelbrotRes = mandelbrotFractal.get_resolution();
    float mandelbrotControlsStartY = static_cast<float>(initialMandelbrotRes.y) + controlPadding;

    tgui::Panel::Ptr controlPanelMandelbrot = tgui::Panel::create();
    controlPanelMandelbrot->setSize({controlWidth + controlPadding * 2, calculate_size_y(290.f)});
    controlPanelMandelbrot->setPosition(controlGroupXOffsetMandelbrot - controlPadding / 2, mandelbrotControlsStartY - controlPadding / 2);
    controlPanelMandelbrot->getRenderer()->setBackgroundColor(sf::Color(240, 240, 240, 200));
    controlPanelMandelbrot->getRenderer()->setBorderColor(sf::Color(150, 150, 150, 255));
    controlPanelMandelbrot->getRenderer()->setBorders(2);
    controlPanelMandelbrot->getRenderer()->setRoundedBorderRadius(5);
    controlPanelMandelbrot->getRenderer()->setPadding(20);
    gui.add(controlPanelMandelbrot);

    tgui::Label::Ptr mandelbrotLabel = tgui::Label::create("Mandelbrot Settings");
    mandelbrotLabel->setPosition(controlGroupXOffsetMandelbrot, mandelbrotControlsStartY);
    mandelbrotLabel->setTextSize( calculate_size_y(16));
    gui.add(mandelbrotLabel);

    tgui::ComboBox::Ptr paletteSelectorMandelbrot = tgui::ComboBox::create();
    paletteSelectorMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(mandelbrotLabel) + controlPadding);
    paletteSelectorMandelbrot->setSize({controlWidth, 24 / (1440.0f / window.getSize().y)});
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
    paletteOffsetSliderMandelbrot->setSize({controlWidth, calculate_size_y(18)});
    paletteOffsetSliderMandelbrot->onValueChange([&](float pos) {
        if (mandelbrotFractal.getPallete() == Palletes::HSV) {
            mandelbrotFractal.SetDegreesOffsetForHSV(static_cast<int>(pos));
            needsMandelbrotRender = true;
        }
    });
    gui.add(paletteOffsetSliderMandelbrot);

    tgui::EditBox::Ptr resolutionPickerMandelbrot = tgui::EditBox::create();
    resolutionPickerMandelbrot->setText(resolution);
    resolutionPickerMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(paletteOffsetSliderMandelbrot) + controlPadding);
    resolutionPickerMandelbrot->setSize({controlWidth, calculate_size_y(24)});
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

    /// Max Iterations Control for Mandelbrot.
    /// Includes a label, edit box, and slider for setting the maximum number
    /// of iterations for the Mandelbrot calculation.
    tgui::Label::Ptr maxIterLabelMandelbrot = tgui::Label::create("Max Iterations:");
    maxIterLabelMandelbrot->setTextSize(calculate_size_y(13));
    maxIterLabelMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(resolutionPickerMandelbrot) + controlPadding * 1.5f);
    gui.add(maxIterLabelMandelbrot);
    unsigned int currentMandelbrotIters = std::max(minIterations, std::min(maxIterations, mandelbrotFractal.get_max_iters()));
    mandelbrotFractal.set_max_iters(currentMandelbrotIters);

    tgui::EditBox::Ptr maxIterEditBoxMandelbrot = tgui::EditBox::create();
    maxIterEditBoxMandelbrot->setDefaultText(std::to_string(currentMandelbrotIters));
    maxIterEditBoxMandelbrot->setText(std::to_string(currentMandelbrotIters));
    maxIterEditBoxMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(maxIterLabelMandelbrot) + labelControlVPadding);
    maxIterEditBoxMandelbrot->setSize({iterEditBoxWidth, calculate_size_y(24)});
    maxIterEditBoxMandelbrot->setInputValidator(tgui::EditBox::Validator::UInt);
    gui.add(maxIterEditBoxMandelbrot);

    tgui::Slider::Ptr maxIterSliderMandelbrot = tgui::Slider::create();
    maxIterSliderMandelbrot->setMinimum(static_cast<float>(minIterations));
    maxIterSliderMandelbrot->setMaximum(static_cast<float>(maxIterations));
    maxIterSliderMandelbrot->setValue(static_cast<float>(currentMandelbrotIters));
    maxIterSliderMandelbrot->setPosition(tgui::bindRight(maxIterEditBoxMandelbrot) + controlPadding, tgui::bindTop(maxIterEditBoxMandelbrot) + (tgui::bindHeight(maxIterEditBoxMandelbrot) - tgui::bindHeight(maxIterSliderMandelbrot)) / 2.f);
    maxIterSliderMandelbrot->setSize({iterSliderWidth, calculate_size_y(18)});
    gui.add(maxIterSliderMandelbrot);

    maxIterSliderMandelbrot->onValueChange([&, maxIterEditBoxMandelbrot](float value) {
        unsigned int intValue = static_cast<unsigned int>(value);
        intValue = std::max(minIterations, std::min(maxIterations, intValue));
        mandelbrotFractal.set_max_iters(intValue);
        if (maxIterEditBoxMandelbrot) {
            maxIterEditBoxMandelbrot->setText(std::to_string(intValue));
            maxIterEditBoxMandelbrot->getRenderer()->setBorderColor(sf::Color::Black);
        }
        needsMandelbrotRender = true;
    });

    /// Callback function to update Mandelbrot iterations from the edit box.
    /// Includes input validation and error handling.
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
                    maxIterEditBoxMandelbrot->getRenderer()->setBorderColor(sf::Color::Black);
                    needsMandelbrotRender = true;
                } else {
                    currentValue = mandelbrotFractal.get_max_iters();
                    maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                    if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                    maxIterEditBoxMandelbrot->getRenderer()->setBorderColor(sf::Color::Yellow);
                }
            } else if (maxIterEditBoxMandelbrot) {
                currentValue = mandelbrotFractal.get_max_iters();
                maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxMandelbrot->getRenderer()->setBorderColor(sf::Color::Black);
            }
        } catch (const std::invalid_argument&) {
            if (maxIterEditBoxMandelbrot) {
                currentValue = mandelbrotFractal.get_max_iters();
                maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxMandelbrot->getRenderer()->setBorderColor(sf::Color::Red);
            }
        } catch (const std::out_of_range&) {
            if (maxIterEditBoxMandelbrot) {
                currentValue = mandelbrotFractal.get_max_iters();
                maxIterEditBoxMandelbrot->setText(std::to_string(currentValue));
                if(maxIterSliderMandelbrot) { maxIterSliderMandelbrot->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxMandelbrot->getRenderer()->setBorderColor(sf::Color::Red);
            }
        }
    };
    maxIterEditBoxMandelbrot->onReturnOrUnfocus(updateMandelbrotIterationsFromEditBox);

    /// Checkbox to toggle dynamic iteration updates for Mandelbrot based on zoom level.
    tgui::CheckBox::Ptr UpdateMaxIterationsMandelbrot = tgui::CheckBox::create("Progressive Iterations");
    UpdateMaxIterationsMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(maxIterEditBoxMandelbrot) + controlPadding + calculate_size_y(24) + controlPadding);
    UpdateMaxIterationsMandelbrot->setSize({calculate_size_x(controlWidth / 16), calculate_size_y(24 / 2)});
    UpdateMaxIterationsMandelbrot->onCheck([&]() {
        if (UpdateMaxIterationsMandelbrot->isChecked()) {
            maxIterEditBoxMandelbrot->setEnabled(false);
            maxIterSliderMandelbrot->setEnabled(false);
            updateIterationsDynamicallyMandelbrot = true;
        } else {
            maxIterEditBoxMandelbrot->setEnabled(true);
            maxIterSliderMandelbrot->setEnabled(true);
            updateIterationsDynamicallyMandelbrot = false;
        }
    });
    UpdateMaxIterationsMandelbrot->onUncheck([&]() {
        maxIterEditBoxMandelbrot->setEnabled(true);
        maxIterSliderMandelbrot->setEnabled(true);
        updateIterationsDynamicallyMandelbrot = false;
    });
    gui.add(UpdateMaxIterationsMandelbrot);

    tgui::Button::Ptr ResetMandelbrot = tgui::Button::create("Reset Mandelbrot");
    ResetMandelbrot->setPosition(controlGroupXOffsetMandelbrot, tgui::bindBottom(maxIterEditBoxMandelbrot) + controlPadding);
    ResetMandelbrot->setSize({controlWidth, calculate_size_y(24)});
    ResetMandelbrot->onPress([&]() {
        mandelbrotFractal.reset();
        maxIterEditBoxMandelbrot->setText("300");
        resolutionPickerMandelbrot->setText("800x600");
        paletteOffsetSliderMandelbrot->setValue(0);
        paletteSelectorMandelbrot->setSelectedItem("HSV");
        UpdateMaxIterationsMandelbrot->setChecked(false);
        maxIterEditBoxMandelbrot->setEnabled(true);
        maxIterSliderMandelbrot->setEnabled(true);


        needsMandelbrotRender = true;
    });
    gui.add(ResetMandelbrot);

    /// ComboBox to switch between different computation contexts (NVRTC, CUDA).
    tgui::ComboBox::Ptr context_switcher_mandelbrot = tgui::ComboBox::create();
    context_switcher_mandelbrot->setPosition({controlGroupXOffsetMandelbrot, tgui::bindBottom(UpdateMaxIterationsMandelbrot) + controlPadding});
    context_switcher_mandelbrot->setSize({controlWidth, calculate_size_y(24)});
    context_switcher_mandelbrot->addMultipleItems({"NVRTC", "CUDA"});
    context_switcher_mandelbrot->setSelectedItem("CUDA");
    context_switcher_mandelbrot->onItemSelect([&](const tgui::String& item){
        if(item == "CUDA") mandelbrotFractal.set_context(context_type::CUDA);
        else if(item == "NVRTC") mandelbrotFractal.set_context(context_type::NVRTC);
    });
    bool cudaOK = mandelbrotFractal.get_isCudaAvailable();
    if(!cudaOK) {
        context_switcher_mandelbrot->setEnabled(false);
        context_switcher_mandelbrot->addItem("CPU");
        context_switcher_mandelbrot->setSelectedItem("CPU");
    }

    gui.add(context_switcher_mandelbrot);

    /// Panel and TextArea for entering custom fractal formulas (NVRTC).
    /// Includes a compile button, progress bar, and error display.
    tgui::Panel::Ptr mandelbrotPanelText = tgui::Panel::create();
    mandelbrotPanelText->setPosition(controlGroupXOffsetMandelbrot - controlPadding / 2, tgui::bindBottom(context_switcher_mandelbrot) + controlPadding * 3);
    mandelbrotPanelText->setSize({controlWidth * 2 + controlPadding, calculate_size_y(250)});
    mandelbrotPanelText->getRenderer()->setBackgroundColor({240, 240, 240, 200});
    mandelbrotPanelText->getRenderer()->setPadding(20);
    mandelbrotPanelText->getRenderer()->setBorders(2);
    mandelbrotPanelText->getRenderer()->setBorderColor({150, 150, 150, 255});
    mandelbrotPanelText->getRenderer()->setRoundedBorderRadius(10);
    gui.add(mandelbrotPanelText);

    const unsigned int height_of_text_area = calculate_size_y(200);
    tgui::TextArea::Ptr custom_code = tgui::TextArea::create();
    custom_code->setPosition({controlGroupXOffsetMandelbrot, tgui::bindBottom(context_switcher_mandelbrot) + controlPadding * 4});
    custom_code->setSize({controlWidth * 2, height_of_text_area});
    custom_code->setText("new_real = z_real * z_real - z_imag * z_imag + real;\n"
                         "z_imag =  2 * z_real * z_imag + imag;\n");
    custom_code->setTextSize(calculate_size_y(14));
    custom_code->setReadOnly(!mandelbrotFractal.get_isCudaAvailable());
    gui.add(custom_code);


    tgui::RichTextLabel::Ptr errorText = tgui::RichTextLabel::create();
    errorText->setPosition({tgui::bindRight(custom_code) + controlPadding, tgui::bindTop(custom_code) + controlPadding / 2});
    errorText->setVisible(false);
    errorText->setSize({calculate_size_x(controlWidth * 2), height_of_text_area});
    errorText->getRenderer()->setBackgroundColor({255, 255, 255, 255});
    errorText->getRenderer()->setBorderColor({255, 0, 0, 255});
    errorText->getRenderer()->setBorders(4);
    gui.add(errorText);

    tgui::ProgressBar::Ptr compilationProgressBar = tgui::ProgressBar::create();
    compilationProgressBar->setPosition({controlGroupXOffsetMandelbrot + controlWidth + controlPadding / 2, tgui::bindBottom(custom_code) + controlPadding});
    compilationProgressBar->setSize({calculate_size_x(controlWidth - 5), calculate_size_y(24)});
    compilationProgressBar->setVisible(false);
    gui.add(compilationProgressBar);

    tgui::Label::Ptr compilationErrorPopUp = tgui::Label::create();
    sf::Clock tooltipTimer;
    sf::Time tooltipDisplayDuration = sf::seconds(5);
    compilationErrorPopUp->getRenderer()->setBackgroundColor(sf::Color(255, 255, 255, 255));
    compilationErrorPopUp->getRenderer()->setTextColor(sf::Color::Red);
    compilationErrorPopUp->getRenderer()->setBorders(1);
    compilationErrorPopUp->getRenderer()->setBorderColor(sf::Color(255, 0, 0, 255));
    compilationErrorPopUp->setTextSize(calculate_size_x(16));
    compilationErrorPopUp->setPosition({tgui::bindRight(custom_code) + controlPadding, tgui::bindTop(custom_code)});
    compilationErrorPopUp->setText("<- Compilation Error!");
    compilationErrorPopUp->setVisible(false);
    gui.add(compilationErrorPopUp);

    /// Button to trigger compilation of the custom fractal formula.
    /// Updates progress bar and displays compilation errors or success messages.
    auto parse  = tgui::Button::create("Compile!");
    parse->setPosition({controlGroupXOffsetMandelbrot, tgui::bindBottom(custom_code) + controlPadding});
    parse->setSize({controlWidth, calculate_size_y(24)});
    parse->onPress([&](){
        parse->getRenderer()->setBorderColor(sf::Color::Blue);
        std::string code = custom_code->getText().toStdString();
        compilationProgressBar->setVisible(true);
        gui.draw();
        try {
            std::shared_future<std::string> error = mandelbrotFractal.set_custom_formula(code);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            while(mandelbrotFractal.get_is_compiling()) {
                compilationProgressBar->setValue(mandelbrotFractal.get_compiling_percentage() / 2.0);
                window.clear();
                window.draw(sf::Sprite(renderTarget.getTexture()));
                window.draw(mandelbrotHardnessDisplay);
                window.draw(juliaHardnessDisplay);
                window.draw(fpsDisplay);
                gui.draw();
                window.display();
            }
            if(std::size(error.get()) > 1) {
                tooltipTimer.restart();
                compilationErrorPopUp->setVisible(true);
                errorText->setText("<size=20><b><color=#ff0000>Compilation Error!</color></size>\n" + error.get());
                compilationProgressBar->setValue(0);
                compilationProgressBar->setVisible(false);
                throw std::runtime_error("Compilation error");
            }
            else {
                errorText->setText("<size=20><color=#00aa00>Compilation Successful</color></size>\n");
            }
            compilationProgressBar->setValue(50);
            parse->getRenderer()->setBorderColor(sf::Color::Green);
        }
        catch (const std::exception& e) {
            parse->getRenderer()->setBorderColor(sf::Color::Red);
            std::cerr << "Error: " << e.what() << std::endl;
            return;
        }
        try {
            std::shared_future<std::string> error = juliaFractal.set_custom_formula(code);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            while(juliaFractal.get_is_compiling()) {
                compilationProgressBar->setValue(50 + juliaFractal.get_compiling_percentage() / 2.0);
                window.clear();
                window.draw(sf::Sprite(renderTarget.getTexture()));
                window.draw(mandelbrotHardnessDisplay);
                window.draw(juliaHardnessDisplay);
                window.draw(fpsDisplay);
                gui.draw();
                window.display();
            }
            if(std::size(error.get()) > 1) {
                tooltipTimer.restart();
                compilationErrorPopUp->setVisible(true);
                errorText->setText("<size=20><b><color=#ff0000>Compilation Error!</color></size>\n" + error.get());
                compilationProgressBar->setValue(0);
                compilationProgressBar->setVisible(false);
                throw std::runtime_error("Compilation error");
            }
            else {
                errorText->setText("<size=20><color=#00aa00><b>Compilation Successful</b></color></size>\n");
            }
            compilationProgressBar->setValue(100);
            parse->getRenderer()->setBorderColor(sf::Color::Green);
        }
        catch (const std::exception& e) {
            parse->getRenderer()->setBorderColor(sf::Color::Red);
            std::cerr << "Error: " << e.what() << std::endl;
            return;
        }

        needsMandelbrotRender = true;
        needsJuliaRender = true;
        context_switcher_mandelbrot->setSelectedItem("NVRTC");
        compilationProgressBar->setVisible(false);
    });
    gui.add(parse);

    /// Button to toggle compilation error text area.
    sf::Image icon({10, 10}, sf::Color::Black);
    if(!icon.loadFromFile(R"(Images/Info.png)")) std::cerr << "Error loading icon!" << std::endl;
    tgui::BitmapButton::Ptr errorButton = tgui::BitmapButton::create();
    errorButton->setImage(sf::Texture(icon));
    errorButton->setSize({18, 18});
    errorButton->getRenderer()->setBackgroundColor({255, 255, 255, 0});
    errorButton->getRenderer()->setBorderColor({255, 255, 255, 0});
    errorButton->getRenderer()->setBorders(0);
    errorButton->setPosition({tgui::bindRight(custom_code) - 25, tgui::bindTop(custom_code) + 2});
    errorButton->onPress([&](){
        if(compilationErrorPopUp->isVisible()) { compilationErrorPopUp->setVisible(false); }
        if(errorText->isVisible()) {
            errorText->setVisible(false);
        } else {
            errorText->setVisible(true);
        }
    });
    gui.add(errorButton);


    /// Julia control panel and widgets.
    /// Sets up UI elements for configuring the Julia fractal,
    /// including palette selection, HSV offset, resolution, max iterations,
    /// reset button, progressive iteration checkbox, and animation toggle.
    sf::Vector2i initialJuliaRes = juliaFractal.get_resolution();
    float juliaControlsStartY = static_cast<float>(initialJuliaRes.y) + controlPadding;

    tgui::Panel::Ptr juliaPanel = tgui::Panel::create();
    juliaPanel->setPosition(controlGroupXOffsetJulia - controlPadding, juliaControlsStartY - controlPadding / 2);
    juliaPanel->setSize({controlWidth + controlPadding * 2, calculate_size_y(300)});
    juliaPanel->getRenderer()->setBackgroundColor({240, 240, 240, 200});

    juliaPanel->getRenderer()->setPadding(500);
    juliaPanel->getRenderer()->setBorders(2);
    juliaPanel->getRenderer()->setBorderColor({150, 150, 150, 255});
    juliaPanel->getRenderer()->setRoundedBorderRadius(10);
    gui.add(juliaPanel);


    tgui::Label::Ptr juliaLabel = tgui::Label::create("Julia Settings");
    juliaLabel->setPosition(controlGroupXOffsetJulia, juliaControlsStartY + controlPadding);
    juliaLabel->setTextSize(16);
    gui.add(juliaLabel);


    tgui::Slider::Ptr paletteOffsetSliderJulia = tgui::Slider::create();
    paletteOffsetSliderJulia->setMinimum(0);
    paletteOffsetSliderJulia->setMaximum(360);
    paletteOffsetSliderJulia->setValue(0);
    paletteOffsetSliderJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(juliaLabel) + controlPadding * 2 + ( calculate_size_y(24)));
    paletteOffsetSliderJulia->setSize({controlWidth, calculate_size_y(18)});
    paletteOffsetSliderJulia->onValueChange([&](float pos) {
        if (juliaFractal.getPallete() == Palletes::HSV || juliaFractal.getPallete() == Palletes::CyclicHSV) {
            juliaFractal.SetDegreesOffsetForHSV(static_cast<int>(pos));
            needsJuliaRender = true;
        }
    });
    gui.add(paletteOffsetSliderJulia);

    tgui::ComboBox::Ptr paletteSelectorJulia = tgui::ComboBox::create();
    paletteSelectorJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(juliaLabel) + controlPadding);
    paletteSelectorJulia->setSize({controlWidth, calculate_size_y(24)});
    paletteSelectorJulia->setTextSize(calculate_size_y(12));
    ComboBoxCreator(paletteSelectorJulia, paletteNames, "HSV");
    paletteSelectorJulia->onItemSelect([&](const tgui::String& str) {
        std::string paletteName = str.toStdString();
        if (paletteName != "HSV") paletteOffsetSliderJulia->setEnabled(false);
        else paletteOffsetSliderJulia->setEnabled(true);
        juliaFractal.setPallete(str.toStdString());
        needsJuliaRender = true;
    });
    gui.add(paletteSelectorJulia);


    tgui::EditBox::Ptr resolutionPickerJulia = tgui::EditBox::create();
    resolutionPickerJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(paletteOffsetSliderJulia) + controlPadding);
    resolutionPickerJulia->setSize({controlWidth, calculate_size_y(24)});
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
                    juliaHardnessDisplay.setPosition({ static_cast<float>(window.getSize().x - juliaFractal.get_resolution().x + 10), 40.f });
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

    /// Max Iterations Control for Julia.
    /// Includes a label, edit box, and slider for setting the maximum number
    /// of iterations for the Julia calculation.
    tgui::Label::Ptr maxIterLabelJulia = tgui::Label::create("Max Iterations:");
    maxIterLabelJulia->setTextSize(calculate_size_y(13));
    maxIterLabelJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(resolutionPickerJulia) + controlPadding * 1.5f);
    gui.add(maxIterLabelJulia);
    unsigned int currentJuliaIters = std::max(minIterations, std::min(maxIterations, juliaFractal.get_max_iters()));
    juliaFractal.set_max_iters(currentJuliaIters);

    tgui::EditBox::Ptr maxIterEditBoxJulia = tgui::EditBox::create();
    maxIterEditBoxJulia->setDefaultText(std::to_string(currentJuliaIters));
    maxIterEditBoxJulia->setText(std::to_string(currentJuliaIters));
    maxIterEditBoxJulia->setTextSize(calculate_size_y(12));
    maxIterEditBoxJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(maxIterLabelJulia) + labelControlVPadding);
    maxIterEditBoxJulia->setSize({iterEditBoxWidth, calculate_size_y(24)});
    maxIterEditBoxJulia->setInputValidator(tgui::EditBox::Validator::UInt);
    gui.add(maxIterEditBoxJulia);

    tgui::Slider::Ptr maxIterSliderJulia = tgui::Slider::create();
    maxIterSliderJulia->setMinimum(static_cast<float>(minIterations));
    maxIterSliderJulia->setMaximum(static_cast<float>(maxIterations));
    maxIterSliderJulia->setValue(static_cast<float>(currentJuliaIters));
    maxIterSliderJulia->setPosition(tgui::bindRight(maxIterEditBoxJulia) + controlPadding, tgui::bindTop(maxIterEditBoxJulia) + (tgui::bindHeight(maxIterEditBoxJulia) - tgui::bindHeight(maxIterSliderJulia)) / 2.f);
    maxIterSliderJulia->setSize({iterSliderWidth, calculate_size_y(18)});
    gui.add(maxIterSliderJulia);

    maxIterSliderJulia->onValueChange([&, maxIterEditBoxJulia](float value) {
        unsigned int intValue = static_cast<unsigned int>(value);
        intValue = std::max(minIterations, std::min(maxIterations, intValue));
        juliaFractal.set_max_iters(intValue);
        if (maxIterEditBoxJulia) {
            maxIterEditBoxJulia->setText(std::to_string(intValue));
            maxIterEditBoxJulia->getRenderer()->setBorderColor("black");
        }
        needsJuliaRender = true;
    });

    /// Callback function to update Julia iterations from the edit box.
    /// Includes input validation and error handling.
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
                    maxIterEditBoxJulia->getRenderer()->setBorderColor({0, 0, 0, 255});
                    needsJuliaRender = true;
                } else {
                    currentValue = juliaFractal.get_max_iters();
                    maxIterEditBoxJulia->setText(std::to_string(currentValue));
                    if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                    maxIterEditBoxJulia->getRenderer()->setBorderColor({255, 165, 0, 255});
                }
            } else if (maxIterEditBoxJulia) {
                currentValue = juliaFractal.get_max_iters();
                maxIterEditBoxJulia->setText(std::to_string(currentValue));
                if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxJulia->getRenderer()->setBorderColor({0, 0, 0, 255});
            }
        } catch (const std::invalid_argument&) {
            if (maxIterEditBoxJulia) {
                currentValue = juliaFractal.get_max_iters();
                maxIterEditBoxJulia->setText(std::to_string(currentValue));
                if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxJulia->getRenderer()->setBorderColor({255, 0, 0, 255});
            }
        } catch (const std::out_of_range&) {
            if (maxIterEditBoxJulia) {
                currentValue = juliaFractal.get_max_iters();
                maxIterEditBoxJulia->setText(std::to_string(currentValue));
                if(maxIterSliderJulia) { maxIterSliderJulia->setValue(static_cast<float>(currentValue)); }
                maxIterEditBoxJulia->getRenderer()->setBorderColor({255, 0, 0, 255});
            }
        }
    };
    maxIterEditBoxJulia->onReturnOrUnfocus(updateJuliaIterationsFromEditBox);

    /// Toggle button to start/stop a progressive animation of Julia iterations.
    tgui::ToggleButton::Ptr UpdateIterationsJulia = tgui::ToggleButton::create("Start Cool Animation");
    unsigned int startingIterations = 300;
    UpdateIterationsJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(maxIterEditBoxJulia) + controlPadding + calculate_size_y(24) + controlPadding);
    UpdateIterationsJulia->setSize({controlWidth, calculate_size_y(24)});
    UpdateIterationsJulia->onToggle([&](){
        if(progressiveAnimation)
            juliaFractal.set_max_iters(startingIterations);
        else
            startingIterations = juliaFractal.get_max_iters();
        progressiveAnimation = !progressiveAnimation;
    });
    UpdateIterationsJulia->setTextSize(calculate_size_y(14));
    gui.add(UpdateIterationsJulia);

    /// Checkbox to toggle dynamic iteration updates for Julia based on zoom level.
    tgui::CheckBox::Ptr UpdateMaxIterationsJulia = tgui::CheckBox::create("Progressive Iterations");
    UpdateMaxIterationsJulia->setTextSize(calculate_size_y(13));
    UpdateMaxIterationsJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(UpdateIterationsJulia) + controlPadding);
    UpdateMaxIterationsJulia->setSize({controlWidth / 16, calculate_size_y(24 / 2)});
    UpdateMaxIterationsJulia->onCheck([&]() {
        if (UpdateMaxIterationsJulia->isChecked()) {
            maxIterEditBoxJulia->setEnabled(false);
            maxIterSliderJulia->setEnabled(false);
            updateIterationsDynamicallyJulia = true;
        } else {
            maxIterEditBoxJulia->setEnabled(true);
            maxIterSliderJulia->setEnabled(true);
            updateIterationsDynamicallyJulia = false;
        }
    });
    UpdateMaxIterationsJulia->onUncheck([&]() {
        maxIterEditBoxJulia->setEnabled(true);
        maxIterSliderJulia->setEnabled(true);
        updateIterationsDynamicallyJulia = false;
    });
    gui.add(UpdateMaxIterationsJulia);

    tgui::RichTextLabel::Ptr helpText = tgui::RichTextLabel::create();
    helpText->setSize(window.getSize().x, window.getSize().y);
    helpText->setPosition(0, 0);
    helpText->getRenderer()->setBackgroundColor({0, 0, 0, 240});
    helpText->setText("<size=60><color=#00aa00>Controls:</color></size>\n\n"
                      "<color=#00aa00>Left Mouse Button</color> - Drag Mandelbrot/Julia\n\n"
                      "<color=#00aa00>Right Mouse Button</color> - Draw Iteration Lines\n\n"
                      "<color=#00aa00>WASD</color> - Move Mandelbrot\n\n"
                      "<color=#00aa00>Scroll Wheel</color> - Zoom In/Out\n\n"
                      "<color=#00aa00>F6</color> - Take Screenshot\n\n"
                      "<color=#00aa00>B</color> - Block Julia Updates\n\n"
                      "<color=#00aa00>T</color> - Start/Stop Timelapse\n\n"
                      "<color=#00aa00>H</color> - Open this menu\n\n"
                      "<size=40><color=#00aa00>Esc</color> - Close Application</size>\n\n");
    helpText->setTextSize(calculate_size_x(40));
    helpText->setVisible(false);
    helpText->getRenderer()->setTextColor(sf::Color::White);
    gui.add(helpText);

    tgui::RichTextLabel::Ptr nvrtcFormattingText = tgui::RichTextLabel::create();
    nvrtcFormattingText->setPosition(0, 0);
    nvrtcFormattingText->setSize(window.getSize().x, window.getSize().y);
    nvrtcFormattingText->getRenderer()->setBackgroundColor({0, 0, 0, 0});
    nvrtcFormattingText->setText("\n<size=40><color=#00aa00>NVRTC Formatting:</color></size>\n"
                                 "This is a very basic Introduction to NVRTC code writing.\n"
                                 "1. You should follow basic C++ syntax rules.\n\n"
                                 "2. You can use the following variables:\n"
                                 "<color=#00aa00>z_real</color> - Real part of the complex number z - \n"
                                 "<color=#00aa00>z_imag</color> - Imaginary part of the complex number z - \n"
                                 " <color=#00aa00>real</color> - Real part of the complex number c - \n"
                                 "<color=#00aa00>imag</color> - Imaginary part of the complex number c - \n"
                                 "<color=#00aa00>new_real</color> - Real part of the new complex number z - \n"
                                 "<color=#00aa00>PI</color> - number PI in float precision - \n"
                                 "<color=#00aa00>GOLDEN_RATIO</color> - Golden ratio in float precision - \n"
                                 "<color=#00aa00>z_comp</color> - Current distance from center of complex plane -\n"
                                 "3. If you want to use hardcoded variables in your code, write them as following:\n\n"
                                 "T(1.0) - in this case the compiler will use the current precision rendered at.\n"
                                 "Otherwise may be used unwanted precision with following long renderers or corrupted images\n\n"
                                 "4. You may use any function you know about, simply type it's name and provide it with required parameters (examples):\n"
                                 "sin(T(10)), "
                                 "log(T(20)), "
                                 "exp(T(42)).\n"
                                 "If you wanna use faster version of that function (less precision), use the following syntax:\n"
                                 "__sinf(T(10)), "
                                 "__logf(T(20)), "
                                 "so on...\n\n"
                                 "Thanks for Reading that quick introduction to compiling formulas syntax, for any more details read FORMULA_WRITING_SYNTAX in project files, have fun!\n");
    nvrtcFormattingText->setHorizontalAlignment(tgui::HorizontalAlignment::Right);
    nvrtcFormattingText->setTextSize(calculate_size_x(25));
    nvrtcFormattingText->getRenderer()->setTextColor(sf::Color::White);
    nvrtcFormattingText->setVisible(false);
    gui.add(nvrtcFormattingText);


    tgui::Button::Ptr ResetJulia = tgui::Button::create("Reset Julia");
    ResetJulia->setPosition(controlGroupXOffsetJulia, tgui::bindBottom(maxIterEditBoxJulia) + controlPadding);
    ResetJulia->setSize({controlWidth, calculate_size_y(24)});
    ResetJulia->onPress([&]() {
        juliaFractal.reset();
        juliaFractal.setPallete("HSV");
        paletteOffsetSliderJulia->setValue(0);
        paletteOffsetSliderJulia->setEnabled(true);
        maxIterEditBoxJulia->setText("300");
        maxIterSliderJulia->setValue(300);
        resolutionPickerJulia->setText("800x600");
        UpdateMaxIterationsJulia->setChecked(false);
        needsJuliaRender = true;
        UpdateIterationsJulia->setDown(false);
    });

    ResetJulia->setTextSize(calculate_size_y(14));
    gui.add(ResetJulia);


    mandelbrotFractal.set_resolution({int(windowSize.size.x / 2), int(windowSize.size.y)});
    juliaFractal.set_resolution({int(windowSize.size.x / 2), int(windowSize.size.y)});

    resolutionPickerMandelbrot->setText(std::to_string(mandelbrotFractal.get_resolution().x) + "x" + std::to_string(mandelbrotFractal.get_resolution().y));
    resolutionPickerJulia->setText(std::to_string(juliaFractal.get_resolution().x) + "x" + std::to_string(juliaFractal.get_resolution().y));
    resolutionPickerMandelbrot->setTextSize(calculate_size_y(12));
    resolutionPickerJulia->setTextSize(calculate_size_y(12));


    /// Main application loop.
    /// Processes events, updates state, renders fractals, and draws the frame.
    window.setFramerateLimit(360);
    while (window.isOpen()) {
        /// Event processing.
        /// Handles user input (mouse, keyboard) and window events.
        /// Events are first processed by TGUI, then custom logic handles
        /// fractal interaction (dragging, zooming, parameter changes).
        /// User interaction resets the quality improvement timer.
        while (const auto event = window.pollEvent()) {
            gui.handleEvent(*event);

            if (const auto* mouseMoveEvent = event->getIf<sf::Event::MouseMoved>()) {
                mousePosition = mouseMoveEvent->position;

                /// Check if mouse is over UI panels to prevent fractal interaction
                /// when controls are being used.
                bool mouseOverAnyLabel = isMouseInsideAnyLabelManual(mousePosition, {juliaPanel, controlPanelMandelbrot, mandelbrotPanelText});

                IsInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < mandelbrotFractal.get_resolution().x &&
                                     mousePosition.y >= 0 && mousePosition.y < mandelbrotFractal.get_resolution().y &&
                                     !mouseOverAnyLabel;

                IsInJuliaArea = mousePosition.x >= window.getSize().x - juliaFractal.get_resolution().x && mousePosition.x < window.getSize().x &&
                                mousePosition.y >= 0 && mousePosition.y < juliaFractal.get_resolution().y &&
                                !mouseOverAnyLabel;

                if (IsInMandelbrotArea) {
                    mouseMovedInMandelbrotArea = true;
                    if (showMandelbrotIterationLines) {
                        mandelbrotFractal.drawIterationLines(mousePosition);
                    }
                    if (mandelbrotFractal.get_is_dragging()) {
                        currentMandelbrotQuality = RenderQuality::good;
                        mandelbrotFractal.dragging(mousePosition);
                    }
                }
                else if (IsInJuliaArea) {
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

            /// Keyboard event handling.
            /// Handles specific key presses for actions like closing the window,
            /// blocking Julia updates, moving the Mandelbrot view, toggling
            /// timelapse, taking screenshots, and forcing re-renders.
            if (const auto* keyPressEvent = event->getIf<sf::Event::KeyPressed>()) {
                IsInMandelbrotArea = mousePosition.x >= 0 && mousePosition.x < mandelbrotFractal.get_resolution().x && mousePosition.y >= 0 && mousePosition.y < mandelbrotFractal.get_resolution().y;

                switch(keyPressEvent->scancode) {
                    case sf::Keyboard::Scancode::Escape:
                        window.close();
                        break;

                    case sf::Keyboard::Scancode::B:
                        if (IsInMandelbrotArea) {
                            blockJuliaParameterUpdate = !blockJuliaParameterUpdate;
                            if (!blockJuliaParameterUpdate) needsJuliaRender = true;
                        }
                        break;

                    case sf::Keyboard::Scancode::W:
                    case sf::Keyboard::Scancode::S:
                    case sf::Keyboard::Scancode::A:
                    case sf::Keyboard::Scancode::D:
                        if (IsInMandelbrotArea) {
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
                        } else {
                            juliaFractal.stop_timelapse();
                            needsJuliaRender = true;
                            currentJuliaQuality = RenderQuality::best;
                        }
                        break;

                    case sf::Keyboard::Scancode::F6:
                    {
                        sf::Texture texture;
                        texture = sf::Texture(sf::Vector2u(mandelbrotFractal.getTexture().getSize().x, mandelbrotFractal.getTexture().getSize().y), true);
                        if (true) {
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
                            filename += ".jpg";

                            if (screenshot.saveToFile(filename)) {
                                std::cout << "Screenshot saved to " << filename << std::endl;
                            } else {
                                std::cerr << "Error: Failed to save screenshot to " << filename << std::endl;
                            }
                        }
                        break;
                    }
                    case sf::Keyboard::Scancode::F12:
                        if (IsInMandelbrotArea) {
                            needsMandelbrotRender = true;
                            needsJuliaRender = true;
                        }
                        break;
                    case sf::Keyboard::Scancode::H:
                        if (updateScreen) {
                            helpText->setVisible(true);
                            nvrtcFormattingText->setVisible(true);
                            window.clear();
                            window.draw(sf::Sprite(renderTarget.getTexture()));
                            window.draw(mandelbrotHardnessDisplay);
                            window.draw(juliaHardnessDisplay);
                            window.draw(fpsDisplay);
                            gui.draw();
                            window.display();
                            updateScreen = false;

                        } else {
                            helpText->setVisible(false);
                            nvrtcFormattingText->setVisible(false);
                            updateScreen = true;
                        }
                        break;
                }
            }

            /// Mouse wheel event handling for zooming.
            /// Zooms the fractal under the mouse cursor. Triggers 'good' quality render.
            /// Dynamically adjusts iterations if the corresponding checkbox is checked.
            if (const auto* mouseWheelEvent = event->getIf<sf::Event::MouseWheelScrolled>()) {
                float delta = mouseWheelEvent->delta;
                if (IsInMandelbrotArea) {
                    isZoomingMandelbrot = true;
                    currentMandelbrotQuality = RenderQuality::good;
                    mandelbrotFractal.handleZoom(delta, mousePosition);
                    updateIterationsDynamicallyMandelbrot? mandelbrotFractal.set_max_iters( (unsigned int)std::min( std::max( 300.0 / std::sqrt(240.0) * std::sqrt(std::max(1.0, mandelbrotFractal.get_zoom_x())), 50.0), 10000.0 )) : mandelbrotFractal.set_max_iters(mandelbrotFractal.get_max_iters());
                }
                else if (IsInJuliaArea) {
                    isZoomingJulia = true;
                    currentJuliaQuality = RenderQuality::good;
                    juliaFractal.handleZoom(delta, {int(mousePosition.x - window.getSize().x + juliaFractal.get_resolution().x), mousePosition.y});
                    updateIterationsDynamicallyJulia? juliaFractal.set_max_iters( (unsigned int)std::min( std::max( 300.0 / std::sqrt(240.0) * std::sqrt(std::max(1.0, juliaFractal.get_zoom_x())), 50.0), 10000.0 )) : juliaFractal.set_max_iters(juliaFractal.get_max_iters());
                    needsJuliaRender = true;
                }
            }

            /// Mouse button pressed event handling.
            /// Starts dragging for left click, shows iteration lines for right click.
            /// Triggers 'good' quality render.
            if (const auto* mouseButtonPressedEvent = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressedEvent->button == sf::Mouse::Button::Left) {
                    if (IsInMandelbrotArea) {
                        isLeftMouseDownMandelbrot = true;
                        currentMandelbrotQuality = RenderQuality::good;
                        mandelbrotFractal.start_dragging(mousePosition);
                    } else if (IsInJuliaArea) {
                        isLeftMouseDownJulia = true;
                        currentJuliaQuality = RenderQuality::good;
                        juliaFractal.start_dragging(mousePosition);
                    }
                }
                else if (mouseButtonPressedEvent->button == sf::Mouse::Button::Right) {
                    if (IsInMandelbrotArea) {
                        showMandelbrotIterationLines = true;
                        mandelbrotFractal.drawIterationLines(mousePosition);
                        needsMandelbrotRender = true;
                    }
                }
            }

            /// Mouse button released event handling.
            /// Stops dragging, hides iteration lines.
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
                    if (IsInMandelbrotArea) {
                        showMandelbrotIterationLines = false;
                    }
                }
            }

            /// Window resize handling.
            /// Note: Resizing is currently disabled by resetting the window size
            /// back to its previous dimensions.
            if (const auto pResized = event->getIf<sf::Event::Resized>()) {
                window.setSize(oldWindowSize);
            }
            else {
                oldWindowSize = window.getSize();
            }
        }

        if(!updateScreen) continue;

        /// Check and hide compilation error tooltip after a delay.
        if (compilationErrorPopUp->isVisible() && tooltipTimer.getElapsedTime() > tooltipDisplayDuration) {
            tooltipTimer.restart();
            compilationErrorPopUp->setVisible(false);
        }

        /// Calculate delta time and update frame counter for FPS calculation.
        sf::Time deltaTime = frameClock.restart();
        frameCountForFPS++;

        /// Update Mandelbrot idle time.
        /// If interacting (dragging/zooming), reset idle time. Otherwise, accumulate.
        bool isDraggingMandelbrotThisFrame = isLeftMouseDownMandelbrot && mouseMovedInMandelbrotArea;
        bool isZoomingMandelbrotThisFrame = isZoomingMandelbrot;
        bool isInteractingMandelbrotThisFrame = isDraggingMandelbrotThisFrame || isZoomingMandelbrotThisFrame;

        if (isInteractingMandelbrotThisFrame) {
            mandelbrotIdleTimeMs = 0.0f;
        } else {
            mandelbrotIdleTimeMs += deltaTime.asMilliseconds();
        }

        /// Mandelbrot rendering decision logic.
        /// Determines if a render is needed based on explicit flags,
        /// user interaction (dragging/zooming), or sufficient idle time
        /// to trigger a 'best' quality render.
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
        }

        /// Update Julia idle time.
        /// If interacting (dragging/zooming), reset idle time. Otherwise, accumulate.
        bool isDraggingJuliaThisFrame = isLeftMouseDownJulia && mouseMovedInJuliaArea;
        bool isZoomingJuliaThisFrame = isZoomingJulia;
        bool isInteractingJuliaThisFrame = isDraggingJuliaThisFrame || isZoomingJuliaThisFrame;

        if (isInteractingJuliaThisFrame) {
            juliaIdleTime = sf::Time::Zero;
        } else {
            juliaIdleTime += deltaTime;
        }

        /// Julia rendering decision logic.
        /// Determines if a render is needed based on explicit flags,
        /// user interaction, timelapse mode, progressive animation,
        /// mouse position over the Mandelbrot set (to update Julia parameters),
        /// or sufficient idle time for a 'best' quality render.
        bool renderJuliaThisFrame = false;
        RenderQuality qualityForJuliaRender = currentJuliaQuality;
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
        else if (mouseMovedInMandelbrotArea && !blockJuliaParameterUpdate && IsInMandelbrotArea)
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
                if (needsImprovement && !IsInMandelbrotArea && (currentJuliaQuality != RenderQuality::best || previousJuliaQuality != RenderQuality::best))
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
            juliaFractal.setPosition({ static_cast<float>(window.getSize().x - juliaFractal.get_resolution().x), 0.f });
        }

        /// Drawing phase.
        /// Clears the render target and window, draws the fractals to the
        /// render target, displays the render target on the window,
        /// and finally draws the TGUI elements and text overlays.
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

        /// Update FPS and hardness display periodically.
        if (fpsUpdateClock.getElapsedTime().asSeconds() > 0.2f) {
            float elapsedSeconds = fpsUpdateClock.restart().asSeconds();
            float fps = frameCountForFPS / elapsedSeconds;
            fpsDisplay.setString("FPS: " + std::to_string(fps));
            mandelbrotHardnessDisplay.setString("M-Hardness: " + std::to_string(mandelbrotFractal.get_hardness_coeff()));
            juliaHardnessDisplay.setString("J-Hardness: " + std::to_string(juliaFractal.get_hardness_coeff()));
            frameCountForFPS = 0;
        }

        /// Reset per-frame interaction flags.
        mouseMovedInMandelbrotArea = false;
        isZoomingMandelbrot = false;
        mouseMovedInJuliaArea = false;
        isZoomingJulia = false;

    }
    return 0;
}