#include <TGUI/Backend/SFML-Graphics.hpp>
#include <TGUI/Core.hpp>
#include <TGUI/AllWidgets.hpp>
#include "CUDA files/FractalClass.cuh"
#include <iostream>
#include <thread>
#include <chrono>

void main_thread() {
	// --- Interaction State Variables ---
    enum InteractionType { NONE, ZOOM, OTHER };
    static InteractionType last_interaction = NONE;
    static float time_since_interaction_end = 0.0f;
    static float time_since_interaction_end_julia = 0.0f;
    const float ZOOM_DELAY = 1000;
    sf::Clock clock;

    // --- Render State Variables ---
    render_state mandelbrot_quality = render_state::good;
    render_state previous_mandelbrot_quality = render_state::good;
    render_state julia_quality = render_state::good;
    render_state previous_julia_quality = render_state::good;

    // --- Render Flags ---
    bool should_render_mandelbrot = true;
    bool should_render_julia = true;
    bool should_render_iteration_lines = false;

    // --- SFML Window and Buffer ---
    sf::RenderWindow window(sf::VideoMode({ 1920, 1080 }), "Fractals");
    sf::RenderTexture render_buffer({ 1920, 1080 });

    // --- Font and Text Elements ---
    sf::Font font;
    if (!font.openFromFile("C:\\Windows\\Fonts\\ariblk.ttf")) {
        return; // Handle font loading error appropriately, maybe throw an exception or log error
    }
    sf::Text fps_text(font);
    fps_text.setPosition({ 10, 10 });
    fps_text.setString("0");
    sf::Text mandelbrot_hardness_text(font);
    mandelbrot_hardness_text.setPosition({ 10, 40 });
    mandelbrot_hardness_text.setString("0.0");
    sf::Text julia_hardness_text(font);
    julia_hardness_text.setPosition({ 1920 - 800 + 10, 40 }); // Position relative to Julia set window
    julia_hardness_text.setString("0.0");

    // --- Mouse Interaction Variables ---
    sf::Vector2i initial_mouse_pos; // Base mouse position, might not be used
    sf::Vector2i current_mouse_pos;

    double julia_zx = 0, julia_zy = 0; // Complex coordinates for Julia set calculation

    bool is_drawing_julia = false;
    bool is_mouse_pressed = false;
    bool is_first_mouse_move = true;
    bool is_dragging_fractal = false;
    bool timelapse_rendering = false;
    bool is_zooming = false;
    bool is_mouse_in_area = false;
    bool is_mouse_pressed_in_area = is_mouse_pressed && is_mouse_in_area;
    bool is_interacting = is_zooming || is_mouse_pressed_in_area;

    bool is_zooming_julia = false;
    bool is_mouse_in_arear_julia = false;
    bool is_mouse_pressed_in_area_julia = false;
    bool is_interacting_julia = is_zooming_julia || is_mouse_pressed_in_area;
    bool should_improve_quality = false;
    bool should_improve_quality_julia = false;
    bool is_mouse_pressed_julia = false;
    bool has_mouse_moved_julia_area = false;
    sf::Time time_since_julia_interaction_end;

    // --- Timers and Clock ---
    sf::Clock mandelbrot_render_timer;
    sf::Clock julia_render_timer;
    sf::Clock fps_timer;
    sf::Clock quality_improvement_timer;

    float current_fps = 0.0f;

    // --- Fractal Objects ---
    FractalBase<fractals::mandelbrot> mandelbrot_set;
    FractalBase<fractals::julia> julia_set;
    julia_set.setPosition({ static_cast<float>(window.getSize().x - 800), 0.f }); // Position Julia set to the right
    double max_iterations = mandelbrot_set.get_max_iters();

    bool block_julia_updates = false;

    bool has_mouse_moved_mandelbrot_area = true;

    window.setFramerateLimit(360);

    // --- Performance Measurement ---
    float gflops = measureGFLOPS(50000);
    std::cout << "Gflops: " << gflops << "\n";
    mandelbrot_set.setMaxComputation(gflops);
    julia_set.setMaxComputation(gflops);

    // --- TGUI Setup ---
    tgui::Gui gui(window);

    tgui::Slider::Ptr palette_slider = tgui::Slider::create();
    palette_slider->setMinimum(0);
    palette_slider->setMaximum(360);
    palette_slider->onValueChange([&](float pos) {
        if (mandelbrot_set.getPallete() == Palletes::HSV)
            mandelbrot_set.SetDegreesOffsetForHSV(static_cast<int>(pos));
        if (julia_set.getPallete() == Palletes::HSV)
            julia_set.SetDegreesOffsetForHSV(static_cast<int>(pos));
        should_render_mandelbrot = true;
        should_render_julia = true;
        });
    palette_slider->setPosition({ 810, 40 });

    tgui::ComboBox::Ptr palette_combobox = tgui::ComboBox::create();
    palette_combobox->setPosition({ 810, 10 });
    palette_combobox->addItem("HSV");
    palette_combobox->addItem("BlackOWhite");
    palette_combobox->addItem("Basic");
    palette_combobox->addItem("OscillatingGrayscale");
    palette_combobox->addItem("Fire");
    palette_combobox->addItem("Pastel");
    palette_combobox->addItem("Interpolated");
    palette_combobox->addItem("CyclicHSV");
    palette_combobox->addItem("FractalPattern");
    palette_combobox->addItem("PerlinNoise");
    palette_combobox->addItem("Water");
    palette_combobox->addItem("Sunset");
    palette_combobox->addItem("DeepSpace");
    palette_combobox->addItem("Physchodelic");
    palette_combobox->addItem("IceCave");
    palette_combobox->addItem("ElectricNebula");
    palette_combobox->addItem("AccretionDisk");

    palette_combobox->setSelectedItem("HSV");
    palette_combobox->onItemSelect([&](tgui::String str) {
        should_render_mandelbrot = true;
        mandelbrot_set.setPallete(std::string(str));
        should_render_julia = true;
        julia_set.setPallete(std::string(str));
        });

    gui.add(palette_combobox);
    gui.add(palette_slider);

    // --- Main Render Loop ---
    while (window.isOpen()) {
        // --- Event Handling ---
        while (const auto event = window.pollEvent()) {
            if (const auto gg = event->getIf<sf::Event::MouseWheelScrolled>() || event->is<sf::Event::MouseMoved>() || event->is<sf::Event::KeyPressed>() || event->is<sf::Event::MouseButtonPressed>()) {
                quality_improvement_timer.restart();
            }

            if (const auto* mouse_move_event = event->getIf<sf::Event::MouseMoved>()) {
                current_mouse_pos = mouse_move_event->position;

                if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                    if (should_render_iteration_lines) {
                        mandelbrot_set.drawIterationLines(current_mouse_pos);
                    }
                    has_mouse_moved_mandelbrot_area = true;
                }
                if (current_mouse_pos.x > 1920 - 800 && current_mouse_pos.y < 600) {
                    julia_set.drawIterationLines(current_mouse_pos);
                }
            }

            if (event->is<sf::Event::Closed>()) {
                window.close();
            }

            if (const auto* key_press_event = event->getIf<sf::Event::KeyPressed>()) {
                if (key_press_event->scancode == sf::Keyboard::Scancode::Escape) {
                    window.close();
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::R) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600)
                        mandelbrot_set.set_max_iters(mandelbrot_set.get_max_iters() * 2);
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::B) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600)
                        block_julia_updates = !block_julia_updates;
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::W) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        should_render_mandelbrot = true;
                        mandelbrot_quality = render_state::good;
                        mandelbrot_set.move_fractal({ 0, 1 });
                    }
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::S) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        should_render_mandelbrot = true;
                        mandelbrot_quality = render_state::good;
                        mandelbrot_set.move_fractal({ 0, -1 });
                    }
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::A) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        should_render_mandelbrot = true;
                        mandelbrot_quality = render_state::good;
                        mandelbrot_set.move_fractal({ 1, 0 });
                    }
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::D) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        should_render_mandelbrot = true;
                        mandelbrot_quality = render_state::good;
                        mandelbrot_set.move_fractal({ -1, 0 });
                    }
                }
                else if (key_press_event->scancode == sf::Keyboard::Scancode::T) {
                    if(timelapse_rendering) julia_set.stop_timelapse();
                    else julia_set.start_timelapse();
                    timelapse_rendering = !timelapse_rendering;
                }
            }

            if (const auto* mouse_wheel_event = event->getIf<sf::Event::MouseWheelScrolled>()) {
                if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                    should_render_mandelbrot = true;
                    is_zooming = true;
                    mandelbrot_quality = render_state::good;
                    mandelbrot_set.handleZoom(mouse_wheel_event->delta, current_mouse_pos);
                }
                else if (current_mouse_pos.x > window.getSize().x - 800 && current_mouse_pos.y < 600) {
                    should_render_julia = true;
                    is_zooming_julia = true;
                    julia_quality = render_state::good;
                    julia_set.handleZoom(mouse_wheel_event->delta, current_mouse_pos);
                }
            }

            if (const auto* mouse_button_pressed_event = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouse_button_pressed_event->button == sf::Mouse::Button::Left) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        is_mouse_pressed = true;
                        mandelbrot_quality = render_state::good;
                        mandelbrot_set.start_dragging(current_mouse_pos);
                    }
                    else if (current_mouse_pos.x > window.getSize().x - 800 && current_mouse_pos.y < 600) {
                        is_mouse_pressed_julia = true;
                        julia_set.start_dragging(current_mouse_pos);
                    }
                }
                if (mouse_button_pressed_event->button == sf::Mouse::Button::Right) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        should_render_iteration_lines = true;
                        mandelbrot_set.drawIterationLines(current_mouse_pos);
                    }
                }
            }

            if (const auto* mouse_button_released_event = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouse_button_released_event->button == sf::Mouse::Button::Left) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        is_mouse_pressed = false;
                        mandelbrot_quality = render_state::best;
                        mandelbrot_set.stop_dragging();
                    }
                    else if (current_mouse_pos.x > window.getSize().x - 800 && current_mouse_pos.y < 600) {
                        is_mouse_pressed_julia = false;
                        julia_set.stop_dragging();
                    }
                }
                else if (mouse_button_released_event->button == sf::Mouse::Button::Right) {
                    if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                        should_render_iteration_lines = false;
                    }
                }
            }

            if (const auto* mouse_drag_event = event->getIf<sf::Event::MouseMoved>()) {
                if (current_mouse_pos.x < 800 && current_mouse_pos.y < 600) {
                    has_mouse_moved_mandelbrot_area = true;
                    if (mandelbrot_set.get_is_dragging()) {
                        should_render_mandelbrot = true;
                        mandelbrot_quality = render_state::good;
                        mandelbrot_set.dragging({ current_mouse_pos.x, current_mouse_pos.y });
                    }
                }
                else if (current_mouse_pos.x > window.getSize().x - 800 && current_mouse_pos.y < 600 && julia_set.get_is_dragging()) {
                    has_mouse_moved_julia_area = true;
                    if (julia_set.get_is_dragging()) {
                        julia_set.dragging({ current_mouse_pos.x, current_mouse_pos.y });
                        should_render_julia = true;
                        julia_quality = render_state::good;
                    }
                }
            }
            gui.handleEvent(*event);
        } // --- End Event Handling ---
        sf::Time delta_time = clock.restart();
        ++current_fps;

        is_dragging_fractal = has_mouse_moved_mandelbrot_area && is_mouse_pressed;
        is_zooming = is_zooming && (time_since_interaction_end += delta_time.asMilliseconds()) > 500;
        is_interacting = is_dragging_fractal || is_zooming;

        if (is_interacting) {
            time_since_interaction_end = 0;
            should_improve_quality = false;
        }
        else {
            time_since_interaction_end += delta_time.asMilliseconds();
        }

        bool can_improve_quality = time_since_interaction_end > 500;

        if (is_interacting) {
            mandelbrot_quality = render_state::good;
        }
        else if (can_improve_quality) {
            mandelbrot_quality = render_state::best;
        }

        bool should_render =
            should_render_mandelbrot ||
            is_interacting ||
            (can_improve_quality && !is_interacting && mandelbrot_quality != render_state::best);


        if (should_render)
        {
            should_render_mandelbrot = false;
            previous_mandelbrot_quality = mandelbrot_quality;
            std::string quality_str = mandelbrot_quality == render_state::best ? "Best" : (mandelbrot_quality == render_state::good ? "Good" : "Bad");
            mandelbrot_set.render(mandelbrot_quality);

            auto start_draw_time = std::chrono::high_resolution_clock::now();
            render_buffer.clear();
            window.clear();
            render_buffer.draw(mandelbrot_set);
            if (is_drawing_julia) {
                render_buffer.draw(julia_set);
            }

            render_buffer.display();
            window.draw(sf::Sprite(render_buffer.getTexture()));
            window.draw(fps_text);
            window.draw(mandelbrot_hardness_text);
            window.draw(julia_hardness_text);
            gui.draw();

            auto render_time = mandelbrot_render_timer.restart();
            window.display();
            auto end_draw_time = std::chrono::high_resolution_clock::now();
            auto draw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_draw_time - start_draw_time);
            std::cout << "Time needed to apply SSAA4 and draw to window: " << draw_duration.count() << " ms\n";
            std::cout << "Mandelbrot set " << "(" << quality_str << ")" << " was drawn in : " << render_time.asMilliseconds() << " ms" << std::endl;
            std::cout << "Zoom scale: " << mandelbrot_set.get_zoom_x() << "\n";
            should_render_mandelbrot = false;
            if (should_improve_quality)
                previous_mandelbrot_quality = static_cast<render_state>(static_cast<int>(previous_mandelbrot_quality) + 1);
        } // --- End Mandelbrot Rendering ---

        bool is_dragging_fractal_julia = has_mouse_moved_julia_area && is_mouse_pressed_julia;
        is_zooming_julia = is_zooming_julia && (time_since_interaction_end_julia += delta_time.asMilliseconds()) > 500;
        is_interacting_julia = is_dragging_fractal_julia || is_zooming_julia;
        bool is_interacting_julia_this_frame = is_interacting_julia || should_render_julia;

        if (is_interacting_julia_this_frame) {
            time_since_interaction_end_julia = 0;
        }
        else {
            time_since_interaction_end_julia += delta_time.asMilliseconds();
        }

        bool can_improve_julia_quality = time_since_interaction_end_julia > 500;

        render_state qual = julia_quality;

        if (is_interacting_julia_this_frame) {
            julia_quality = render_state::good;
        }
        else if (can_improve_julia_quality) {
            julia_quality = render_state::best;
        }

        bool should_render_julia_this_frame = (
            should_render_julia ||
            is_interacting_julia_this_frame ||
            (can_improve_julia_quality && !is_interacting_julia_this_frame && qual != render_state::best));

        if (timelapse_rendering) {
            julia_set.update_timelapse();

            render_buffer.clear();
            window.clear();

            render_buffer.draw(mandelbrot_set);
            render_buffer.draw(julia_set);

            render_buffer.display();

            window.draw(sf::Sprite(render_buffer.getTexture()));
            window.draw(fps_text);
            window.draw(mandelbrot_hardness_text);
            window.draw(julia_hardness_text);
            gui.draw();

            window.display();
        }

        // --- Julia Rendering (Conditional on Mouse Move in Mandelbrot Area) ---
        else if (has_mouse_moved_mandelbrot_area && !block_julia_updates)
        {
            is_drawing_julia = true;
            julia_zx = -(mandelbrot_set.get_x_offset() - (current_mouse_pos.x / mandelbrot_set.get_zoom_x()));
            julia_zy = -(mandelbrot_set.get_y_offset() - (current_mouse_pos.y / mandelbrot_set.get_zoom_y()));

            julia_set.render(render_state::good, julia_zx, julia_zy);
            julia_set.setPosition({ static_cast<float>(window.getSize().x - 800), 0 });

            render_buffer.clear();
            window.clear();

            render_buffer.draw(mandelbrot_set);
            render_buffer.draw(julia_set);

            render_buffer.display();

            window.draw(sf::Sprite(render_buffer.getTexture()));
            window.draw(fps_text);
            window.draw(mandelbrot_hardness_text);
            window.draw(julia_hardness_text);
            gui.draw();

            window.display();

            auto julia_render_time = julia_render_timer.restart();

            has_mouse_moved_mandelbrot_area = false;
            should_render_julia = false;
        } // --- End Julia Rendering (Conditional on Mouse Move in Mandelbrot Area) ---

        else if (should_render_julia_this_frame)
        {
            should_render_julia = false;
            previous_julia_quality = julia_quality;
            std::string quality_str_julia = julia_quality == render_state::best ? "Best" : "Good";

            julia_set.render(julia_quality, julia_zx, julia_zy);
            julia_set.setPosition({ static_cast<float>(window.getSize().x - 800), 0 });

            render_buffer.clear();
            window.clear();
            render_buffer.draw(mandelbrot_set);
            if (is_drawing_julia) {
                render_buffer.draw(julia_set);
            }

            render_buffer.display();
            window.draw(sf::Sprite(render_buffer.getTexture()));
            window.draw(fps_text);
            window.draw(mandelbrot_hardness_text);
            window.draw(julia_hardness_text);
            gui.draw();
            window.display();


            std::cout << "julia set " << "(" << quality_str_julia << ")\n";
            if (can_improve_julia_quality)
                previous_julia_quality = static_cast<render_state>(static_cast<int>(previous_julia_quality) + 1);
        }


        else {
            // --- No Fractal Render Needed, Just UI ---
            window.clear();
            window.draw(sf::Sprite(render_buffer.getTexture()));
            window.draw(fps_text);
            window.draw(mandelbrot_hardness_text);
            window.draw(julia_hardness_text);
            gui.draw();
            window.display();
        } // --- End No Fractal Render Needed, Just UI --


        // --- FPS Counter Update ---
        if (fps_timer.getElapsedTime().asSeconds() > 0.2f) {
            float elapsed = fps_timer.getElapsedTime().asSeconds();
            fps_text.setString(std::to_string(current_fps / elapsed)); // Cast to int for cleaner FPS display
            mandelbrot_hardness_text.setString(std::to_string(mandelbrot_set.get_hardness_coeff()));
            julia_hardness_text.setString(std::to_string(julia_set.get_hardness_coeff()));
            current_fps = 0;
            fps_timer.restart();
        } // --- End FPS Counter Update ---

        should_render_mandelbrot = should_render_mandelbrot || mandelbrot_quality != render_state::best;

    } // --- End Main Render Loop ---
}

int main() {
    std::thread mainThread(main_thread);
    mainThread.join();
}